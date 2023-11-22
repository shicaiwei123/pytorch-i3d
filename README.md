# pytorch-i3d
pytorch i3d for mutimodal action Recognition (UCF101,HMDB51,NTUD60, NW-UCLA)

## Dataset
- [NW-UCLA](https://ieeexplore.ieee.org/document/6909735/references#references)
- [NTUD60](https://ieeexplore.ieee.org/document/7780484)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

## RUN
- command
    ```bash
       cd src
       bash train_nwucla.sh
       bash train_ntud60.sh
       bash train_hmdb_i3d.sh
       bash train_ucf101_i3d.sh
    ```
- parameters
  - **data_root**: Path to data
  - **clip_len**: Number of used frame
  - **mode**: Modality, rgb, depth, flow

