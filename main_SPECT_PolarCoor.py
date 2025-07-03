import torch
import numpy as np
import time
from scipy.io import loadmat
import argparse
import pickle
from process_list_SPECT_PolarCoor import get_compton_backproj_list
from recon_mlem_SPECT_PolarCoor import run_recon_mlem
import os
import sys
import shutil

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

if __name__ == '__main__':
    with torch.no_grad():
        # file path
        data_file_path = "ContrastPhantom_240_90_1e10"
        factor_file_path = "20250625"

        # set system factors
        e0 = 0.662  # energy of incident photons
        ene_resolution_662keV = 0.1  # energy resolution at 662keV
        ene_resolution = ene_resolution_662keV * (0.662 / e0) ** 0.5
        ene_threshold_max = 0.477
        ene_threshold_min = 0.050

        # fov factors
        pixel_num_layer = 1500
        pixel_num_z = 40
        pixel_num = pixel_num_layer * pixel_num_z
        rotate_num = 10

        # intrinsic spatial resolution of scintillators
        delta_r1 = 1.25
        delta_r2 = 1.25
        alpha = 1

        # divide list-mode data into subsets to prevent GPU overload
        num_workers = 10

        # reconstruction factors
        iter_arg = argparse.ArgumentParser().parse_args()
        iter_arg.sc = 1000  # CntStat only
        iter_arg.jsccd = 500  # List only
        iter_arg.jsccsd = 1000  # CntStat+List
        iter_arg.save_iter_step = 1
        iter_arg.osem_subset_num = 5
        iter_arg.t_divide_num = 10       # prevent memory explosions during iterations
        iter_arg.event_level = 2

        # down sampling ratio of events
        flag_ds = 0
        ds = 1

        # whether to store t
        flag_save_t = 0
        flag_save_s = 0

        # --------Step1: Checking Devices--------
        time_start = time.time()
        # start getting outputs
        logfile = open("print_log.txt", "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, logfile)

        print("")
        print("--------Step1: Checking Devices--------")
        print("Checking Devices starts")
        # judge if CUDA is available and set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available, running on GPU")
        else:
            device = torch.device("cpu")
            print("CUDA is not available, running on CPU")

        print("Checking Devices ends, time used:", time.time() - time_start, "s")

        # --------Step2: Loading Files--------
        print("--------Step2: Loading Files--------")
        print("Loading Files starts")

        # Factors
        sysmat_file_path = "./Factors/" + factor_file_path + "/SysMat_polar"
        detector_file_path = "./Factors/" + factor_file_path + "/Detector.csv"
        coor_polar_file_path = "./Factors/" + factor_file_path + "/coor_polar_full.csv"
        rotmat_file_path = "./Factors/" + factor_file_path + "/RotMat_full.csv"
        rotmat_inv_file_path = "./Factors/" + factor_file_path + "/RotMatInv_full.csv"
        sysmat = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32), [pixel_num, -1])).transpose(0,1)
        detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])
        coor_polar = torch.from_numpy(np.genfromtxt(coor_polar_file_path, delimiter=",", dtype=np.float32))
        rotmat = torch.from_numpy(np.genfromtxt(rotmat_file_path, delimiter=",", dtype=int))
        rotmat_inv = torch.from_numpy(np.genfromtxt(rotmat_inv_file_path, delimiter=",", dtype=int))

        # Data
        list_file_path = "./List/List_" + data_file_path + "/"
        proj_file_path = "./CntStat/CntStat_" + data_file_path + ".csv"
        proj = torch.from_numpy(np.reshape(np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32), [rotate_num, -1])).transpose(0, 1)
        list_origin = []
        for i in range(0, rotate_num):
            list_file_path_tmp = list_file_path + str(i+1) + ".csv"
            list_origin_tmp = torch.from_numpy(np.genfromtxt(list_file_path_tmp, delimiter=",", dtype=np.float32)[:, 0:4])
            list_origin.append(list_origin_tmp)

        print("Loading Files ends, time used:", time.time() - time_start, "s")

        # --------Step3: Data Downsampling--------
        print("--------Step3: Data Downsampling--------")
        print("Data Downsampling starts")

        if flag_ds == 1:
            print("Downsampling On")
            for i in range(0, rotate_num):
                # porj
                proj_tmp = proj[:, i]
                proj_ds_tmp = proj[:, i] * 0
                proj_s_index_tmp = torch.tensor([i for i in range(proj_tmp.size(0)) for _ in range(round(proj_tmp[i].item()))])
                indices_tmp = torch.randperm(proj_s_index_tmp.size(0))
                selected_indices_tmp = indices_tmp[0:int(torch.round(proj_tmp.sum() * ds).item())]
                proj_s_index_ds_tmp = proj_s_index_tmp[selected_indices_tmp]
                for j in range(0, proj_ds_tmp.size(dim=0)):
                    proj_ds_tmp[j] = (proj_s_index_ds_tmp == j).sum()
                proj[:, i] = proj_ds_tmp

                # list
                list_origin_tmp = list_origin[i]
                indices_tmp = torch.randperm(list_origin_tmp.size(0))
                selected_indices_tmp = indices_tmp[0:int(list_origin_tmp.size(0) * ds)]
                list_origin_tmp = list_origin_tmp[selected_indices_tmp, :]
                list_origin[i] = list_origin_tmp
        else:
            print("Downsampling Off")

        print("Data Downsampling ends, time used:", time.time() - time_start, "s")

        # --------Step4: Processing List--------
        print("--------Step4: Processing List--------")
        print("Processing List starts")

        sysmat = sysmat.to(device)
        detector = detector.to(device)
        coor_polar = coor_polar.to(device)

        if flag_save_t == 1:
            t = []
            t_compton = []
            t_single = []
            size_t = 0
            compton_event_count_list = torch.zeros(size=[rotate_num, 1], dtype=torch.int)
            for i in range(0, rotate_num):
                list_origin_tmp = list_origin[i]
                list_origin_tmp_chunks = torch.chunk(list_origin_tmp, num_workers, dim=0)

                t_tmp = []
                t_compton_tmp = []
                t_single_tmp = []
                for list_origin_tmp_chunk in list_origin_tmp_chunks:
                    t_chunk, t_compton_chunk, t_single_chunk = get_compton_backproj_list(list_origin_tmp_chunk.to(device), delta_r1, delta_r2, e0, ene_resolution,
                                                                ene_threshold_max, ene_threshold_min, detector, coor_polar, sysmat, device)
                    t_tmp.append(t_chunk)
                    t_compton_tmp.append(t_compton_chunk)
                    t_single_tmp.append(t_single_chunk)
                    torch.cuda.empty_cache()

                t_tmp = torch.cat(t_tmp, dim=0)
                t_compton_tmp = torch.cat(t_compton_tmp, dim=0)
                t_single_tmp = torch.cat(t_single_tmp, dim=0)
                t.append(t_tmp)
                t_compton.append(t_compton_tmp)
                t_single.append(t_single_tmp)

                compton_event_count_list[i] = t_tmp.size(0)
                size_t = size_t + t_tmp.element_size() * t_tmp.nelement()
                print("Rotate Num", str(i+1), "ends, time used:", time.time() - time_start, "s")

            # create a proj that has an equal count
            proj_d = torch.zeros(size=proj.size(), dtype=torch.float32)
            for i in range(0, rotate_num):
                proj_tmp = proj[:, i]
                proj_s_index_tmp = torch.tensor([i for i in range(proj_tmp.size(0)) for _ in range(round(proj_tmp[i].item()))])
                indices_tmp = torch.randperm(proj_s_index_tmp.size(0))
                selected_indices_tmp = indices_tmp[0:compton_event_count_list[i]]
                proj_d_index_tmp = proj_s_index_tmp[selected_indices_tmp]
                for j in range(0, proj_d.size(dim=0)):
                    proj_d[j, i] = (proj_d_index_tmp == j).sum()

            # save t
            t_save_path = "./Backproj/" + data_file_path + "/JSCC/Polar"
            t_compton_save_path = "./Backproj/" + data_file_path + "/ComptonCone/Polar"
            t_single_save_path = "./Backproj/" + data_file_path + "/SysMat/Polar"
            if not os.path.exists(t_save_path):
                os.makedirs(t_save_path)
                os.makedirs(t_compton_save_path)
                os.makedirs(t_single_save_path)

            for i in range(0, rotate_num):
                rotmat_inv_tmp = rotmat_inv[:, i]

                print("Saving t of Rotate Num", str(i+1))
                # JSCC
                t_save_path_tmp = t_save_path + "/" + str(i+1)
                with open(t_save_path_tmp, "w") as file:
                    t[i][:, rotmat_inv_tmp-1].transpose(0, 1).cpu().numpy().astype('float32').tofile(file)

                # Compton Cone
                t_compton_save_path_tmp = t_compton_save_path + "/" + str(i + 1)
                with open(t_compton_save_path_tmp, "w") as file:
                    t_compton[i][:, rotmat_inv_tmp-1].transpose(0, 1).cpu().numpy().astype('float32').tofile(file)

                # SysMat
                t_single_save_path_tmp = t_single_save_path + "/" + str(i + 1)
                with open(t_single_save_path_tmp, "w") as file:
                    t_single[i][:, rotmat_inv_tmp-1].transpose(0, 1).cpu().numpy().astype('float32').tofile(file)

        else:
            t = []
            size_t = 0
            compton_event_count_list = torch.zeros(size=[rotate_num, 1], dtype=torch.int)
            for i in range(0, rotate_num):
                list_origin_tmp = list_origin[i]
                list_origin_tmp_chunks = torch.chunk(list_origin_tmp, num_workers, dim=0)
                t_tmp = []
                for list_origin_tmp_chunk in list_origin_tmp_chunks:
                    t_chunk, _, _ = get_compton_backproj_list(list_origin_tmp_chunk.to(device), delta_r1, delta_r2, e0, ene_resolution,
                                                                ene_threshold_max, ene_threshold_min, detector, coor_polar, sysmat, device)
                    t_tmp.append(t_chunk)
                    torch.cuda.empty_cache()

                t_tmp = torch.cat(t_tmp, dim=0)
                t.append(t_tmp)
                compton_event_count_list[i] = t_tmp.size(0)
                size_t = size_t + t_tmp.element_size() * t_tmp.nelement()
                print("Rotate Num", str(i+1), "ends, time used:", time.time() - time_start, "s")

            # create a proj that has an equal count
            proj_d = torch.zeros(size=proj.size(), dtype=torch.float32)
            for i in range(0, rotate_num):
                proj_tmp = proj[:, i]
                proj_s_index_tmp = torch.tensor([i for i in range(proj_tmp.size(0)) for _ in range(round(proj_tmp[i].item()))])
                indices_tmp = torch.randperm(proj_s_index_tmp.size(0))
                selected_indices_tmp = indices_tmp[0:compton_event_count_list[i]]
                proj_d_index_tmp = proj_s_index_tmp[selected_indices_tmp]
                for j in range(0, proj_d.size(dim=0)):
                    proj_d[j, i] = (proj_d_index_tmp == j).sum()

        sysmat = sysmat.cpu()
        del detector, coor_polar

        single_event_count = round(proj.sum().item())
        compton_event_count = round(proj_d.sum().item())
        print("single events = ", single_event_count, ", Compton events = ", compton_event_count)
        print("The size of t is ", size_t / (1024 **3), " GB")
        print("Processing List ends, time used:", time.time() - time_start, "s")

        # --------Step5: Image Reconstruction--------
        print("--------Step5: Image Reconstruction--------")
        print("Image Reconstruction List starts")

        # calculate sensitivity map
        s_map_arg = argparse.ArgumentParser().parse_args()
        s_map_arg.s = torch.zeros(size=[1, sysmat.size(1)], dtype=torch.float32)
        for i in range(0, rotate_num):
            rotmat_inv_tmp = rotmat_inv[:, i]
            s_map_arg.s = s_map_arg.s + torch.sum(sysmat[:, rotmat_inv_tmp - 1], dim=0, keepdim=True).cpu()
        s_map_arg.s = s_map_arg.s.transpose(0, 1)
        s_map_arg.d = s_map_arg.s * compton_event_count / single_event_count

        # save sensitivity
        if flag_save_s == 1:
            with open("sensitivity_polar", "w") as file:
                s_map_arg.s.cpu().numpy().astype('float32').tofile(file)
        torch.cuda.empty_cache()

        save_path = "./Figure/" + data_file_path + "_" + str(ds) + "_Delta" + str(delta_r1) + "_ER" + str(ene_resolution_662keV) + "_OSEM" + str(iter_arg.osem_subset_num) + "_ITER" + str(iter_arg.jsccsd) + "_SDU" + str(single_event_count) + "_DDU" + str(compton_event_count) + "/"
        save_path_polar = save_path + "Polar/"

        run_recon_mlem(sysmat, rotmat, rotmat_inv, proj, proj_d, t, iter_arg, s_map_arg, alpha, save_path_polar, device)

        print("Image Reconstruction ends, time used:", time.time() - time_start, "s")
        print("Total time used:", time.time() - time_start)

        # get all outputs
        logfile.close()
        sys.stdout = sys.__stdout__
        shutil.move("print_log.txt", save_path + "print_log.txt")