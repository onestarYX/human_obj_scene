from pathlib import Path

if __name__ == '__main__':
    in_dir = Path('data/kitti360/drive_0000_250_300')
    out_dir = in_dir.with_name(in_dir.name + '_cars')
    in_rgb_dir = in_dir / 'rgb'
    in_sem_dir = in_dir / 'semantic'
    