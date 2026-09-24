---
license: other
license_name: lucy-edit-dev-model-non-commercial-license
license_link: >-
  https://drive.google.com/file/d/1pX34A-UOEl9CErMUZKdKzhoWhtSI1TJK/view?usp=drive_link
pipeline_tag: video-to-video
library_name: diffusers
---

# Lucy Edit Dev (5B)

<p align="center">
  <img src="assets/logo.png" width="680" alt="Lucy Edit Dev Logo"/>
</p>

<p align="center">
  🧪 <a href="https://github.com/DecartAI/lucy-edit-comfyui"><b>GitHub</b></a>
  &nbsp;|&nbsp; 📖 <a href="https://platform.decart.ai">Playground</a>
  &nbsp;|&nbsp; 📑 <a href="https://d2drjpuinn46lb.cloudfront.net/Lucy_Edit__High_Fidelity_Text_Guided_Video_Editing.pdf">Technical Paper</a>
  &nbsp;|&nbsp; 💬 <a href="https://discord.gg/decart">Discord</a>
</p>

---
<div align="center">

<table>
<tr>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/painter_gothic_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>Put the woman in gothic black jeans and leather jacket and crop top under it.</em>
</td>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/painter_clown_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>1.2) Put her in a clown outfit.</em>
</td>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/painter_bikini_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>1.3) Put the woman in a red bikini with an open thick coat above it.</em>
</td>
</tr>
</table>
</div>


**Lucy Edit Dev** is an open-weight **video editing** model that performs **instruction-guided edits** on videos using free-text prompts — it supports a variety of edits, such as **clothing & accessory changes**, **character changes**, **object insertions**, and **scene replacements** while preserving the motion and composition perfectly.

- 🚀 **First open-source instruction-guided video editing model**
- 🧩 **Built on Wan2.2 5B architecture** — inherits high-compression VAE + DiT stack, making adapting existing scripts and workflows easy.
- 🏃‍♂️ **Motion Preservation** - preserves the motion and composition of videos perfectly, allowing precise edits.
- 🎯 **Edit reliability** — edits are more robust when compared to common inference time methods.
- 🧢 **Wardrobe & accessories** — change outfits, add glasses/earrings/hats/etc.
- 🧌 **Character Changes** — replace characters with monsters, animals and known characters. (e.g., "Replace the person with a polar bear")
- 🗺️ **Scenery swap** — move the scene (e.g., "transform the scene into a 2D cartoon,")  
- 📝 **Pure text instructions** — no finetuning, no masks required for common edits  

ℹ️ Model size: **~5B params**. Build on top of **Wan2.2 5B**.  

---

## 🎬 Demos

<div align="center">
### Sample 1
<table>
<tr>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/man_jacket_alien_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>1.1) Turn the man into an alien</em>
</td>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/man_jacket_polar_bear_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>1.2) Turn the man into a bear</em>
</td>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/man_jacket_snow_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>1.3) Make it snowy</em>
</td>
</tr>
</table>

### Sample 2
<table>
<tr>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/boat_harley_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>2.1) Turn the woman into Harley Quinn</em>
</td>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/boat_lego_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>2.2) Turn the woman into Lego</em>
</td>
<td align="center">
  <video width="100%" controls>
    <source src="https://d2drjpuinn46lb.cloudfront.net/boat_mu_jersy_edit.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br/>
  <em>2.3) Turn the shirt into a sports jersey</em>
</td>
</tr>
</table>

</div>

Note: The prompts above are not enriched, the model will react better to enriched prompts - as described in the prompt guideline section below.

---

## 🔥 Latest News
- **[2025-09-18]**: Initial **Lucy Edit Dev** weights & reference code released.
- **[2025-09-16]**: Diffusers integration PR opened and merged. <a href="https://github.com/huggingface/diffusers/pull/12340">PR #12340</a>.

---


## 🛠️ Quickstart

### Installation

```bash
pip install git+https://github.com/huggingface/diffusers
```

### Inference

Please refer to the "Prompting Guidelines & Supported Edits" section for the best experience.
```python
from typing import List

import torch
from PIL import Image

from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import export_to_video, load_video


# Arguments
url = "https://d2drjpuinn46lb.cloudfront.net/painter_original_edit.mp4"
prompt = "Change the apron and blouse to a classic clown costume: satin polka-dot jumpsuit in bright primary colors, ruffled white collar, oversized pom-pom buttons, white gloves, oversized red shoes, red foam nose; soft window light from left, eye-level medium shot, natural folds and fabric highlights."
negative_prompt = ""
num_frames = 81
height = 480
width = 832

# Load video
def convert_video(video: List[Image.Image]) -> List[Image.Image]:
    video = load_video(url)[:num_frames]
    video = [video[i].resize((width, height)) for i in range(num_frames)]
    return video

video = load_video(url, convert_method=convert_video)

# Load model
model_id = "decart-ai/Lucy-Edit-Dev"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = LucyEditPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Generate video
output = pipe(
    prompt=prompt,
    video=video,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0
).frames[0]

# Export video
export_to_video(output, "output.mp4", fps=24)
```

---

## Prompting Guidelines & Supported Edits

Lucy Edit is built for **precise, realistic, and identity-preserving video edits.**  
Prompts with ~20–30 descriptive words work best. Using the right **trigger words** helps the model understand your intent.  


### Trigger Words
- **Change** → Clothing or color modifications  
- **Add** → Adding animals or objects  
- **Replace** → Object substitution or subject swap  
- **Transform to** → Global scene or style transformations  


### Supported Edit Types

#### 1. Clothing Changes  
✅ **Best performance.** Lucy Edit excels at swapping outfits while preserving motion, pose, and identity.  
*Example*: *“Change the shirt to a kimono with wide sleeves and patterned fabric.”*  


#### 2. Human/Character Replacement  
✅ **Strong results.** Works well for transforming people into new characters or creatures. Detailed prompts are key.  
*Example*: *“Replace the person with a tiger, striped orange fur, muscular build, and glowing green eyes.”*  
*Example*: *“Replace the person with an 2D anime character, big eyes, blue gown and battle scars.”*  


#### 3. Replace Objects  
✅ **Reliable for structure-preserving swaps.** Ideal when replacing one object with another of similar scale.  
*Example*: *“Replace the apple with a glowing crystal ball emitting blue light.”*  


#### 4. Color Changes  
⚠️ **Mixed reliability.** Sometimes subtle, sometimes exaggerated. Works best with precise descriptions.  
*Example*: *“Change the jacket color to deep red leather with a glossy finish.”*  


#### 5. Add Objects  
⚠️ **Often attaches to the subject.** Works best for wearable or handheld props.  
*Example*: *“Add a golden crown on the person’s head, decorated with ornate jewels.”*  


#### 6. Global Transformations  
⚠️ **Effective for backgrounds or scene-wide changes, might alter the subject** Alter environment or style, might, Often changes the identity of the subject.
*Example*: *“Transform the sunny beach into a snowy tundra with falling snowflakes.”*  


### Additional Notes
- **Strengths:** Lucy Edit excels at **identity conservation, edit precision, realism, and prompt adherence.**  
- **Detail matters:** Longer prompts (20–30 words) describing style, appearance, and context improve results.  
- **Frame count:** 81-frame generations produce better temporal consistency than shorter clips.  
 

---

## 📦 Integrations

* ☁️ **Hosted API:** You can access the model on our API and get 5000 free credits <a href="">here</a>.
* 🧨 **Diffusers:** *Coming soon*
* 🧩 **ComfyUI:** *Coming soon*


## 🧭 Roadmap

* ✅ Public Batch API.
* ✅ Diffusers pipeline (`LucyEditPipeline`)
* ✅ Remote ComfyUI custom nodes.
* ✅ Technical Report
* [ ] Local Inference ComfyUI Nodes.
* [ ] LoRA and fine-tuning scripts.

---


## 📣 Citation

```bibtex
@article{decart2025lucyedit,
  title   = {Lucy Edit: Open-Weight Text-Guided Video Editing},
  author  = {DecartAI Team},
  year    = {2025}
  url     = { https://d2drjpuinn46lb.cloudfront.net/Lucy_Edit__High_Fidelity_Text_Guided_Video_Editing.pdf}
 }
```

---

## 🙏 Acknowledgements

Lucy Edit Dev builds on the excellent foundations of **Wan2.2** (5B), and thanks the broader open-source community including **diffusers** and **Hugging Face**.

---

## 📬 Contact

* GitHub Issues: <a href="https://github.com/DecartAI/lucy-edit-comfyui">DecartAI/lucy-edit</a>.
* Discord: Join our discord server, <a href="https://discord.gg/decart">here</a>.