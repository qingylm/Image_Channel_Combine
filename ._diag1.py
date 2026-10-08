from PIL import Image
import glob, os

paths = [
 r"C:\Users\qingylm\AppData\Local\Temp\dsh-B3Yccu\gradio\2c1ea8b8b9d94017a3439fb7dfadfcf01fdb8811235ec5e50ba10e13b7c5c092\ganges_river_pebbles_rough_2k.png",
 r"C:\Users\qingylm\AppData\Local\Temp\dsh-B3Yccu\gradio\89f894f0aba13b10eae136924d0d67a8a12abcd79346f1368f57ff39fdeda05e\ganges_river_pebbles_ao_2k.png",
 r"C:\Users\qingylm\AppData\Local\Temp\dsh-B3Yccu\gradio\bed7155f5d6f9584d12f38f842bf1273991f19efed9fd37922977534c3ec181b\leaf_scattered_gravel_rough_2k.png",
 r"C:\Users\qingylm\AppData\Local\Temp\dsh-B3Yccu\gradio\f5df297784a4a747411ef3f56d18fc74d0bbcb20ff7bc101f1001cb944db3b9a\ganges_river_pebbles_rough_2k.png",
 r"C:\Users\qingylm\AppData\Local\Temp\dsh-B3Yccu\gradio\f5df297784a4a747411ef3f56d18fc74d0bbcb20ff7bc101f1001cb944db3b9a\ganges_river_pebbles_ao_2k.png",
]
for p in paths:
    if not os.path.exists(p):
        print("MISSING", p); continue
    im = Image.open(p)
    info = f"{os.path.basename(p):34s} {os.path.getsize(p)/1e6:6.2f}MB mode={im.mode:6s} size={im.size} info={ {k:v for k,v in im.info.items() if k in ('dpi','compression','mode')} }"
    print(info)
    try:
        ex = im.getextrema()
        print("    extrema:", ex)
    except Exception as e:
        print("    extrema err", e)
    exif_dpi = im.info.get("dpi")
    if exif_dpi: print("    dpi:", exif_dpi)
