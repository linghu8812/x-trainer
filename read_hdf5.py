import h5py


def inspect_hdf5(file_path: str, show_data_sample: bool = False):
    """
    遍历HDF5文件，打印所有组、数据集信息，查看压缩类型
    :param file_path: hdf5文件路径
    :param show_data_sample: 是否打印数据集前少量样本（图像数据很大建议关闭False）
    """
    def visitor(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {name}/")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}[Dataset] {name}")
            print(f"{indent}  shape: {obj.shape}, dtype: {obj.dtype}")
            # 获取压缩信息
            compression = obj.compression
            compression_opts = obj.compression_opts
            print(f"{indent}  compression: {compression}, opts: {compression_opts}")

            # 判断是否LZF压缩
            if compression == "lzf":
                print(f"{indent}  ⚠️ WARNING: This dataset uses LZF compression (the one causing web read error!)")

            # 可选：打印前几个元素样本，图像数据不要开！
            if show_data_sample:
                if obj.size < 500:
                    data = obj[:]
                    print(f"{indent}  sample data:\n{data}")
                else:
                    print(f"{indent}  large dataset, skip print full data")
        print()

    with h5py.File(file_path, "r") as f:
        print(f"===== HDF5 File: {file_path} =====")
        print(f"File open mode: READ\n")
        f.visititems(visitor)


if __name__ == "__main__":
    # ========== 修改这里，换成你的hdf5路径 ==========
    HDF5_FILE = "datasets/auto_pick_cube_right.hdf5"
    # show_data_sample=True 会打印数据样例，图片数据集很大时建议False
    inspect_hdf5(HDF5_FILE, show_data_sample=False)
