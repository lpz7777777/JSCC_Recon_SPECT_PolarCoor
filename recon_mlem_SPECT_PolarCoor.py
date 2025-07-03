import torch
import numpy as np
import time
from scipy.io import loadmat
import os
import threading


def mlem_bin_mode(sysmat_list, proj_list, img, s_map, rotmat, rotmat_inv, osem_subset_num, rotate_num):
    # mlem algorithm
    for i in range(osem_subset_num):
        weight = 0 * img
        for j in range(rotate_num):
            img_rotate = torch.index_select(img, 0, rotmat[:, j] - 1)
            weight_tmp = torch.matmul(sysmat_list[i].transpose(0, 1), proj_list[i][:, j].unsqueeze(1) / torch.matmul(sysmat_list[i], img_rotate))
            weight = weight + torch.index_select(weight_tmp, 0, rotmat_inv[:, j] - 1)
        img = img * weight / s_map

    return img


def mlem_list_mode_event_level_1(t_list, img, s_map, rotmat, rotmat_inv, osem_subset_num, rotate_num, t_divide_num):
    # list mode mlem algorithm
    for i in range(osem_subset_num):
        weight = 0 * img
        for j in range(rotate_num):
            for k in range(t_divide_num):
                img_rotate = torch.index_select(img, 0, rotmat[:, j] - 1)
                weight_tmp = (torch.nan_to_num(t_list[i][j][k].transpose(0, 1) / (torch.matmul(t_list[i][j][k], img_rotate).sum(1)), nan=0, posinf=0, neginf=0).sum(1, keepdim=True))
                weight = weight + torch.index_select(weight_tmp, 0, rotmat_inv[:, j] - 1)
        img = img * weight / s_map

    return img


def mlem_list_mode_event_level_2(t_list, img, s_map, rotmat, rotmat_inv, osem_subset_num, rotate_num, t_divide_num, device):
    # list mode mlem algorithm for large size of t
    # cuda stream of loading t from cpu to gpu
    stream_loader = torch.cuda.Stream(device=device)
    t_pinned_list = []

    def pin_next_t(i_curr, j_curr):
        # shift t to pin_memory
        i_pin = i_curr
        j_pin = j_curr

        j_pin = j_pin + 1
        if j_pin >= rotate_num:
            j_pin = j_pin - rotate_num
            i_pin = i_pin + 1
            if i_pin >= osem_subset_num:
                return None

        for k in range(t_divide_num):
            t_pinned_list.append(t_list[i_pin][j_pin][k].pin_memory())

    def load_next_t(i_curr, j_curr, k_curr):
        # t from cpu to gpu
        i_next = i_curr
        j_next = j_curr
        k_next = k_curr + 1

        if k_next >= t_divide_num:
            k_next = k_next - t_divide_num
            j_next = j_next + 1
            if j_next >= rotate_num:
                j_next = j_next - rotate_num
                i_next = i_next + 1
                if i_next >= osem_subset_num:
                    return None

        if i_next + j_next > 0:
            t_next = t_pinned_list[0].to(device, non_blocking=True)
            del t_pinned_list[0]
            return t_next
        else:
            return t_list[i_next][j_next][k_next].to(device, non_blocking=True)

    for i in range(osem_subset_num):
        weight = 0 * img
        for j in range(rotate_num):
            thread_pin = threading.Thread(target=pin_next_t, args=(i, j))
            thread_pin.start()

            if i + j == 0 :
                for k in range(t_divide_num):
                    # calculate id of t_next
                    if i + j + k > 0:
                        torch.cuda.current_stream().wait_stream(stream_loader)
                        t_curr = t_next
                    else:
                        t_curr = t_list[i][j][k]

                    with torch.cuda.stream(stream_loader):
                        t_next = load_next_t(i, j, k)

                    img_rotate = torch.index_select(img, 0, rotmat[:, j] - 1)
                    weight_tmp = (torch.nan_to_num(t_curr.transpose(0, 1) / (torch.matmul(t_curr, img_rotate).sum(1)), nan=0, posinf=0, neginf=0).sum(1, keepdim=True))
                    weight = weight + torch.index_select(weight_tmp, 0, rotmat_inv[:, j] - 1)

            thread_pin.join()
            torch.cuda.empty_cache()

        img = img * weight / s_map

    return img


def mlem_list_mode_event_level_3(t_list, img, s_map, rotmat, rotmat_inv, osem_subset_num, rotate_num, t_divide_num, device):
    # list mode mlem algorithm for large size of t
    # cuda stream of loading t from cpu to gpu
    stream_loader = torch.cuda.Stream(device=device)
    t_pinned_list = [t_list[0][0][1]]

    for i in range(osem_subset_num):
        weight = 0 * img
        for j in range(rotate_num):
            for k in range(t_divide_num):
                def pin_next_t(i_curr, j_curr, k_curr):
                    # shift t to pin_memory
                    i_pin = i_curr
                    j_pin = j_curr
                    k_pin = k_curr + 2

                    if k_pin >= t_divide_num:
                        k_pin = k_pin - t_divide_num
                        j_pin = j_pin + 1
                        if j_pin >= rotate_num:
                            j_pin = j_pin - rotate_num
                            i_pin = i_pin + 1
                            if i_pin >= osem_subset_num:
                                return None

                    t_pinned_list.append(t_list[i_pin][j_pin][k_pin].pin_memory())

                def load_next_t():
                    # t from cpu to gpu
                    if not t_pinned_list:
                        return None
                    else:
                        t_next = t_pinned_list[0].to(device, non_blocking=True)
                        del t_pinned_list[0]
                        return t_next

                # calculate id of t_next
                if i + j + k > 0:
                    torch.cuda.current_stream().wait_stream(stream_loader)
                    thread_pin.join()
                    t_curr = t_next
                else:
                    t_curr = t_list[i][j][k]

                with torch.cuda.stream(stream_loader):
                    t_next = load_next_t()
                thread_pin = threading.Thread(target=pin_next_t, args=(i, j, k))
                thread_pin.start()

                img_rotate = torch.index_select(img, 0, rotmat[:, j] - 1)
                weight_tmp = (torch.nan_to_num(t_curr.transpose(0, 1) / (torch.matmul(t_curr, img_rotate).sum(1)), nan=0, posinf=0, neginf=0).sum(1, keepdim=True))
                weight = weight + torch.index_select(weight_tmp, 0, rotmat_inv[:, j] - 1)
            torch.cuda.empty_cache()

        img = img * weight / s_map

    return img


def mlem_joint_mode_event_level_1(sysmat_list, proj_list, t_list, img, s_map, rotmat, rotmat_inv, alpha, osem_subset_num, rotate_num, t_divide_num):
    for i in range(osem_subset_num):
        weight = 0 * img
        for j in range(rotate_num):
            for k in range(t_divide_num):
                img_rotate = torch.index_select(img, 0, rotmat[:, j] - 1)
                weight_tmp = (2 - alpha) * torch.nan_to_num(t_list[i][j][k].transpose(0, 1) / (torch.matmul(t_list[i][j][k], img_rotate).sum(1)), nan=0, posinf=0, neginf=0).sum(1, keepdim=True) + alpha * torch.matmul(sysmat_list[i].transpose(0, 1), proj_list[i][:, j].unsqueeze(1) / torch.matmul(sysmat_list[i], img_rotate))
                weight = weight + torch.index_select(weight_tmp, 0, rotmat_inv[:, j] - 1)
        img = img * weight / s_map

    return img


def mlem_joint_mode_event_level_3(sysmat_list, proj_list, t_list, img, s_map, rotmat, rotmat_inv, alpha, osem_subset_num, rotate_num, t_divide_num, device):
    # joint mode mlem algorithm for large size of t
    stream_loader = torch.cuda.Stream(device=device)
    t_pinned_list = [t_list[0][0][1]]

    for i in range(osem_subset_num):
        weight = 0 * img
        for j in range(rotate_num):
            for k in range(t_divide_num):
                def pin_next_t(i_curr, j_curr, k_curr):
                    # shift t to pin_memory
                    i_pin = i_curr
                    j_pin = j_curr
                    k_pin = k_curr + 2

                    if k_pin >= t_divide_num:
                        k_pin = k_pin - t_divide_num
                        j_pin = j_pin + 1
                        if j_pin >= rotate_num:
                            j_pin = j_pin - rotate_num
                            i_pin = i_pin + 1
                            if i_pin >= osem_subset_num:
                                return None

                    t_pinned_list.append(t_list[i_pin][j_pin][k_pin].pin_memory())

                def load_next_t():
                    # t from cpu to gpu
                    if not t_pinned_list:
                        return None
                    else:
                        t_next = t_pinned_list[0].to(device, non_blocking=True)
                        del t_pinned_list[0]
                        return t_next

                # calculate id of t_next
                if i + j + k > 0:
                    torch.cuda.current_stream().wait_stream(stream_loader)
                    thread_pin.join()
                    t_curr = t_next
                else:
                    t_curr = t_list[i][j][k]

                with torch.cuda.stream(stream_loader):
                    t_next = load_next_t()
                thread_pin = threading.Thread(target=pin_next_t, args=(i, j, k))
                thread_pin.start()

                img_rotate = torch.index_select(img, 0, rotmat[:, j] - 1)
                weight_tmp = (2 - alpha) * torch.nan_to_num(t_curr.transpose(0, 1) / (torch.matmul(t_curr, img_rotate).sum(1)), nan=0, posinf=0, neginf=0).sum(1, keepdim=True) + alpha * torch.matmul(sysmat_list[i].transpose(0, 1), proj_list[i][:, j].unsqueeze(1) / torch.matmul(sysmat_list[i], img_rotate))
                weight = weight + torch.index_select(weight_tmp, 0, rotmat_inv[:, j] - 1)
            torch.cuda.empty_cache()

        img = img * weight / s_map

    return img


def save_img(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path):
    with open(save_path + "Image_SC", "wb") as file:
        img_sc.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_SCD", "wb") as file:
        img_scd.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCD", "wb") as file:
        img_jsccd.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCSD", "wb") as file:
        img_jsccsd.cpu().numpy().astype('float32').tofile(file)

    with open(save_path + "Image_SC_Iter_%d_%d" % (iter_arg.sc, iter_arg.sc / iter_arg.save_iter_step), "wb") as file:
        img_sc_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_SCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step), "wb") as file:
        img_scd_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step), "wb") as file:
        img_jsccd_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd / iter_arg.save_iter_step), "wb") as file:
        img_jsccsd_iter.cpu().numpy().astype('float32').tofile(file)

    file.close()


def run_recon_mlem(sysmat, rotmat, rotmat_inv, proj, proj_d, t, iter_arg, s_map_arg, alpha, save_path, device):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pixel_num = sysmat.size(1)
    rotate_num = rotmat.size(1)

    # initial image
    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32).to(device, non_blocking=True)
    img_scd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device, non_blocking=True)
    img_jsccd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device, non_blocking=True)
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32).to(device, non_blocking=True)

    img_sc_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_scd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)

    # divide datas into subsets and make data to gpu or pin memory
    t_list = [[None for _ in range(rotate_num)] for _ in range(iter_arg.osem_subset_num)]
    if iter_arg.event_level == 3:
        for i in range(0, rotate_num):
            t_tmp = list(torch.chunk(t[i], iter_arg.osem_subset_num, dim=0))
            for j in range(0, iter_arg.osem_subset_num):
                t_list[j][i] = list(torch.chunk(t_tmp[j], iter_arg.t_divide_num, dim=0))
        t_list[0][0][0] = t_list[0][0][0].to(device, non_blocking=True)
        t_list[0][0][1] = t_list[0][0][1].pin_memory()

    elif iter_arg.event_level == 2:
        for i in range(0, rotate_num):
            t_tmp = list(torch.chunk(t[i], iter_arg.osem_subset_num, dim=0))
            for j in range(0, iter_arg.osem_subset_num):
                t_list[j][i] = list(torch.chunk(t_tmp[j], iter_arg.t_divide_num, dim=0))
                if i == 0 and j == 0:
                    for k in range(0, iter_arg.t_divide_num):
                        t_list[j][i][k] =t_list[j][i][k].to(device, non_blocking=True)
                elif i == 0 and j == 1:
                    for k in range(0, iter_arg.t_divide_num):
                        t_list[j][i][k] =t_list[j][i][k].pin_memory()

    else:
        for i in range(0, rotate_num):
            t_tmp = list(torch.chunk(t[i].to(device), iter_arg.osem_subset_num, dim=0))
            for j in range(0, iter_arg.osem_subset_num):
                t_list[j][i] = list(torch.chunk(t_tmp[j], iter_arg.t_divide_num, dim=0))

    del t
    cpnum_list = torch.arange(0, proj.size(dim=0))
    random_id = torch.randperm(proj.size(dim=0))
    cpnum_list = cpnum_list[random_id]
    cpnum_list = list(torch.chunk(cpnum_list, iter_arg.osem_subset_num, dim=0))

    sysmat_list = []
    proj_list = []
    proj_d_list = []
    for i in range(0, iter_arg.osem_subset_num):
        sysmat_list.append(sysmat[cpnum_list[i], :].to(device, non_blocking=True))
        proj_list.append(proj[cpnum_list[i], :].to(device, non_blocking=True))
        proj_d_list.append(proj_d[cpnum_list[i], :].to(device, non_blocking=True))

    rotmat = rotmat.to(device, non_blocking=True)
    rotmat_inv = rotmat_inv.to(device, non_blocking=True)
    s_map_arg.s = s_map_arg.s.to(device, non_blocking=True)
    s_map_arg.d = s_map_arg.d.to(device, non_blocking=True)

    # do iteration
    time_start = time.time()

    # self-collimation
    print("Self-Collimation MLEM starts")
    id_save = 0
    for id_iter_sc in range(iter_arg.sc):
        img_sc = mlem_bin_mode(sysmat_list, proj_list, img_sc, s_map_arg.s, rotmat, rotmat_inv, iter_arg.osem_subset_num, rotate_num)
        if (id_iter_sc + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[id_save, :] = torch.squeeze(img_sc).cpu()
            id_save += 1
            print("Iteration ", str(id_iter_sc + 1), " ends, time used:", time.time() - time_start, "s")

    print("Self-Collimation MLEM ends, time used:", time.time() - time_start)
    torch.cuda.empty_cache()

    # sc-d
    print("SC-D MLEM starts")
    id_save = 0
    for id_iter_scd in range(iter_arg.jsccd):
        img_scd = mlem_bin_mode(sysmat_list, proj_d_list, img_scd, s_map_arg.s, rotmat, rotmat_inv, iter_arg.osem_subset_num, rotate_num)
        if (id_iter_scd + 1) % iter_arg.save_iter_step == 0:
            img_scd_iter[id_save, :] = torch.squeeze(img_scd).cpu()
            id_save += 1
            print("Iteration ", str(id_iter_scd + 1), " ends, time used:", time.time() - time_start, "s")

    print("SC-D MLEM ends, time used:", time.time() - time_start)
    torch.cuda.empty_cache()

    if iter_arg.event_level == 3:
        # more events
        # jscc-d
        print("JSCC-D MLEM starts")
        id_save = 0
        for id_iter_jsccd in range(iter_arg.jsccd):
            img_jsccd = mlem_list_mode_event_level_3(t_list, img_jsccd, s_map_arg.d, rotmat, rotmat_inv, iter_arg.osem_subset_num, rotate_num, iter_arg.t_divide_num, device)
            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-D MLEM ends, time used:", time.time() - time_start)

        # jscc-sd
        print("JSCC-SD MLEM starts")
        id_save = 0
        for id_iter_jsccsd in range(iter_arg.jsccsd):
            img_jsccsd = mlem_joint_mode_event_level_3(sysmat_list, proj_list, t_list, img_jsccsd, alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d, rotmat, rotmat_inv, alpha, iter_arg.osem_subset_num, rotate_num, iter_arg.t_divide_num, device)
            if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccsd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-SD MLEM ends, time used:", time.time() - time_start)

    elif iter_arg.event_level == 2:
        pass

    else:
        # less events
        # jscc-d
        print("JSCC-D MLEM starts")
        id_save = 0
        for id_iter_jsccd in range(iter_arg.jsccd):
            img_jsccd = mlem_list_mode_event_level_1(t_list, img_jsccd, s_map_arg.d, rotmat, rotmat_inv, iter_arg.osem_subset_num, rotate_num, iter_arg.t_divide_num)
            # img_jsccd = mlem_list_mode_moreevents(t_list, img_jsccd, s_map_arg.d, rotmat, rotmat_inv, iter_arg.osem_subset_num, rotate_num, device)
            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-D MLEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

        # jscc-sd
        print("JSCC-SD MLEM starts")
        id_save = 0
        for id_iter_jsccsd in range(iter_arg.jsccsd):
            img_jsccsd = mlem_joint_mode_event_level_1(sysmat_list, proj_list, t_list, img_jsccsd, alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d, rotmat, rotmat_inv, alpha, iter_arg.osem_subset_num, rotate_num, iter_arg.t_divide_num)
            if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccsd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-SD MLEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

    # save images as binary file to 'Figure'
    save_img(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path)