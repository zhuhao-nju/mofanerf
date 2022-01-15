# README

## 1. data preprocessing

Data preprocessing contains two parts : 

1. align all raw scans to TU model, and then clip them to remove the area blew the shoulder.
2. render the multi-view images

**align and clip**

You should download the raw scans and extract them in `models ` folder. **Note** the structure of the `models` folder should be same as our demo.  and then execute the code blew

```python
python align_clip.py
```

The aligned and clipped model will be saved in `./outputs/clip_models` folder, and then run the code blew to render multi-view images

```
python render.py
```

the results will be saved in `./outputs/images` folder, the structure is 

```
images/
------{id_idx}/
----------{exp_idx}_{exp}/*****.png
```

