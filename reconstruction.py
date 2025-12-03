# Based on https://github.com/EmilienDupont/coinpp
import torch
import coinpp.conversion as conversion
import matplotlib.pyplot as plt
import numpy as np
import time


def reconstruct(data, modulations, model, converter, patcher):
    """Reconstructs a single data point.

    Args:
        data: A single datapoint. E.g. a single image. Shape (channels, *spatial_shape).
        modulations: A single set of modulations of shape (1, latent_dim) or
            (num_patches, latent_dim) if using patching.
        model:
        converter:
        patcher:
    """
    with torch.no_grad():
        if patcher is None:
            coordinates, features = converter.to_coordinates_and_features(data)
            features_recon = model.modulated_forward(coordinates, modulations)
            # print("features_recon", features_recon.shape)
            data_recon = conversion.features2data(features_recon, batched=False)
        else:
            patches, spatial_shape = patcher.patch(data)
            coordinates, features = converter.to_coordinates_and_features(patches)
            # Shape (num_patches, *patch_shape, feature_dim)
            features_recon = model.modulated_forward(coordinates, modulations)

            # Shape (num_patches, feature_dim, *patch_shape)
            patch_data = conversion.features2data(features_recon, batched=True)
            # Unpatch data, to obtain shape (feature_dim, *spatial_shape)
            data_recon = patcher.unpatch(patch_data, data.shape[1:])
    return data_recon


if __name__ == "__main__":
    import argparse
    import os
    import wandb
    import wandb_utils
    from helpers import get_datasets_and_converter
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb_run_path",
        help="Path of wandb run for trained model.",
        type=str,
        default="/BaseModel_S.pth",
    )

    parser.add_argument(
        "--save_dir",
        help="Directory where data and their reconstructions will be saved.",
        type=str,
        default="/reconstructions_path",
    )

    parser.add_argument(
        "--modulation_dataset",
        help="Name of modulation dataset to use to generate reconstructions.",
        type=str,
        default="/modulations_test_3_steps.pt",
    )

    parser.add_argument(
        "--data_indices",
        help="Indices of points in dataset for which original and reconstructions will be saved.",
        nargs="+",
        type=int,
        default=[300],
    )

    args = parser.parse_args()

    # Load modulations
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    modulations = wandb_utils.load_modulations(
        args.wandb_run_path, args.modulation_dataset, device
    )
    start_time = time.time()

    # Load model
    model, model_args, patcher = wandb_utils.load_model(args.wandb_run_path, device)
    # Load dataset
    train_dataset, test_dataset, converter = get_datasets_and_converter(
        model_args, force_no_random_crop=True
    )
    # Check if modulations were created from train or test set
    if "train" in args.modulation_dataset:
        dataset = train_dataset
    elif "test" in args.modulation_dataset:
        dataset = test_dataset

    # Create directory to save reconstructions if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # for idx in args.data_indices:
    idx = 0
    for data0 in dataset:
        # data, filname, ori_data = data0
        filname, data = data0

        data = torch.tensor(data)
        # data = data.float()
        data = data.to(device)
        if patcher is None:
            # If no patching, extract modulations of shape (1, latent_dim)
            mods = modulations[idx : idx + 1]
        else:
            # modulations is a list of tensors of shape (num_patches, latent_dim)
            # for each data point. Therefore extract single tensor of shape
            # (num_patches, latent_dim)
            mods = modulations[idx]
        data_recon = reconstruct(data, mods, model, converter, patcher)
        # 反归一化
        # print(data_recon[0])
        # denorm_data_recon = data_recon * (ori_data.max() - ori_data.min()) + ori_data.min()
        denorm_data_recon = data_recon
        # print('----------', data_recon[0])
        # Save original data and reconstruction
        if converter.data_type == "image":
            # torch.save(data, os.path.join(args.save_dir, f"original_{idx}.npy"))
            # np.save(os.path.join(args.save_dir, f"original_{filname}.npy"), ori_data)
            # print("data_recon", data_recon.shape)
            print("filname, idx:", filname, idx)
            # print(data_recon[0])
            np.save(
                # data_recon, os.path.join(args.save_dir, f"reconstruction_{idx}.npy")
                os.path.join(args.save_dir, f"reconstruction_{filname}.npy"), denorm_data_recon.cpu().numpy()
            )
            np.save(
                # data_recon, os.path.join(args.save_dir, f"reconstruction_{idx}.npy")
                os.path.join(args.save_dir, f"original_{filname}.npy"), data.cpu().numpy()
            )
        elif converter.data_type in ("mri", "era5"):
            # torch.save(ori_data, os.path.join(args.save_dir, f"original_{idx}.pt"))
            torch.save(
                denorm_data_recon, os.path.join(args.save_dir, f"reconstruction_{idx}.pt")
            )
        idx = idx + 1
    end_time = time.time()
    print("重建时间:%.4f秒" % (end_time-start_time))

    # original_data = np.load(f"/home/zsk/PythonProject/Project24/reconstructions_path/original_test-{idx-1}.npy.npy").reshape(32, 32)
    original_data = np.load(f"/reconstructions_path/original_test-{idx-1}.npy.npy").reshape(32, 128)
    print(original_data.shape)
    reconstruction_data = np.load(f"/reconstructions_path/reconstruction_test-{idx-1}.npy.npy").reshape(32, 128)
    print(reconstruction_data.shape)

    # fig0 = plt.figure(figsize=(16, 8))
    # # plt.imshow(files_data[:, :], cmap='seismic', aspect='auto', origin='lower')
    # plt.imshow(original_data, cmap='seismic', aspect='auto', origin='lower')
    # # # 添加横纵坐标刻度
    # # plt.xticks(range(len(ORI_DATA.shape[1])), ORI_DATA.shape[1])  # 替换 x_tick_labels 为你的横坐标刻度
    # # plt.yticks(range(len(ORI_DATA.shape[1])), ORI_DATA.shape[1])  # 替换 y_tick_labels 为你的纵坐标刻度
    #
    # plt.colorbar()
    # # plt.savefig('/home/zsk/ori-1.png', format='png', dpi=300)
    # plt.show()
    #


    # fig1, ax = plt.subplots(2, 1, figsize=(15, 5), sharex='all')  # 绘图
    # ax = ax.flatten()  # 子图展平,将ax 由n*m的axes组展平成1*nm的axes组(二维变一维)
    x_data = torch.arange(0, original_data.shape[1], 1)  # 横坐标为随时间变化的采样次数，30秒采样30000次
    plt.plot(x_data, original_data[23], color='b', linewidth=1)
    plt.plot(x_data, reconstruction_data[23], color='r', linewidth=1)
    # # x_data = np.arange(125000, 185000, 1)                # 横坐标为随时间变化的采样次数，30秒采样30000次
    # ax[0].plot(x_data, original_data[1], color='b', linewidth=0.3)  # 使用数组画图
    # ax[1].plot(x_data, reconstruction_data[1], color='r', linewidth=0.3)  # 使用数组画图
    # # ax[2].plot(x_data, ORI_DATA[8000, :], color='k', linewidth=0.3)  # 使用数组画图
    # # ax[2].set_xlabel('Number of samples')
    # ax[0].set_ylabel('Amplitude')
    # ax[1].set_ylabel('Amplitude')
    # # ax[2].set_ylabel('Amplitude')
    # ax[0].legend(labels=['data'], ncol=1)
    # ax[1].legend(labels=['data'], ncol=1)
    # ax[2].legend(labels=['data'], ncol=1)
    # plt.suptitle(st.traces[0].stats.starttime + 0)
    # plt.suptitle((st.traces[0].stats.starttime + 10, st.traces[file_num*FiberTraces-1].stats.endtime - 20))
    # plt.suptitle(st.traces[file_num*FiberTraces-1].stats.endtime)
    plt.show()


