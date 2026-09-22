#!/usr/bin/env python3
"""Fixed successful/large-change examples, show unscaled images and20x errors."""
from pathlib import Path
import torch
from PIL import Image, ImageDraw
ROOT=Path('/root/viewdit/results/video/cog_transport_validation')
OUT=Path('/root/viewdit/results/video/v3_evidence')


def img(x):
    return Image.fromarray(((x.clamp(-1,1).permute(1,2,0)+1)*127.5).round().byte().numpy())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['cows_geometric1','horsejump-low_geometric1','libby_encoded_brightness']:
        data=torch.load(ROOT/(name+'.pt'),map_location='cpu',weights_only=True)['videos']
        keys=['base','raw','reference','dual','pixel_splice'];frames=[27,28,29,30,31,32]
        sheet=Image.new('RGB',(130+224*len(frames),30+150*len(keys)),'white');draw=ImageDraw.Draw(sheet)
        draw.text((5,5),name+' | 0-based frames | native display',fill='black')
        for i,k in enumerate(keys):
            draw.text((5,40+i*150),k,fill='black')
            for j,t in enumerate(frames):
                sheet.paste(img(data[k][0,:,t]),(130+224*j,50+i*150));draw.text((130+224*j,32+i*150),'frame'+str(t),fill='black')
        sheet.save(OUT/(name+'.png'))
        animation=[]
        for t in range(33):
            panel=Image.new('RGB',(224*len(keys),150),'white');pen=ImageDraw.Draw(panel)
            for i,k in enumerate(keys):
                pen.text((224*i+2,2),k+' f'+str(t),fill='black');panel.paste(img(data[k][0,:,t]),(224*i,22))
            animation.append(panel)
        animation[0].save(OUT/(name+'.gif'),save_all=True,append_images=animation[1:],duration=125,loop=0)
        print(name,flush=True)


if __name__=='__main__':main()
