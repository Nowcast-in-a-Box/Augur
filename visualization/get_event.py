import pandas as pd
import h5py
import os
import numpy as np
import torch
import torch.nn.functional as F


EVENT_LIST = {
    "tornado": "S843733",
    # "heavyrain": "S827031" # + 12
    # S825535" + 0
    # "flashflood": "S823542"
}
# "S843733" Tornado
# "S851158" Heavyrain

OFFSET = 12 * 0

DATA_ROOT = "/data3/SEVIR"

catalog = pd.read_csv(f"{DATA_ROOT}/CATALOG.csv", low_memory=False)

for k, v in EVENT_LIST.items():

    event_id = v

    row = catalog[(catalog["id"] == event_id) & (catalog["img_type"] == "vil")] 

    # offsets = [int(x) for x in row["minute_offsets"].split(":")]
    print(row)

    if len(row) == 0:
        print("Not found !")

        exit()
        
    row = row.iloc[0]

    file_rel_path = row["file_name"]
    file_index = row["file_index"]
    img_type = row["img_type"]

    h5_path = os.path.join(DATA_ROOT, "data", file_rel_path)

    total_length = 24

    try:
        with h5py.File(h5_path, "r") as f:
            full_event = f[img_type][file_index]
            
            full_event = np.transpose(full_event, (2, 0, 1))
            
            event = full_event[0 + OFFSET:total_length + OFFSET, :, :]
            
            event = event[::2, :, :] / 255
            
            
            sequence = event[0: 6, :, :]
            target = event[6: , :, :]
            
            print(sequence.shape, target.shape)
            
            sequence = F.interpolate(torch.from_numpy(sequence).unsqueeze(0), size=128, mode="bilinear")
            target = F.interpolate(torch.from_numpy(target).unsqueeze(0), size=128, mode="bilinear")
            
            torch.save(sequence, f"case_study/{k}/sequence.pt")
            torch.save(target, f"case_study/{k}/target.pt")
            
            
            
    except Exception as e:
        print(e)