if __name__ == "__main__":
    import os
    from os.path import join
    from totalsegmentator.python_api import totalsegmentator
    root_path = "/mnt/disk_2/Zach/AutopetIV/"
    names = [i.split(".nii.gz")[0] for i in sorted(os.listdir(join(root_path, "labelsTr")))]
    for name in names:
        print(name)
        #totalsegmentator()