<h1 align="center">CoRe-BT: Joint Radiology-Pathology Learning
for Multimodal Brain Tumor Typing</h1>
<p align="center">
    <a href="https://www.imageclef.org/2026/medical/mediqa-core">Challenge Description</a> |
    <a href="https://ai4media-bench.aimultimedialab.ro/competitions/6/">Registration</a> |
    <a href="assets/manuscript.pdf">Manuscript</a>
    <!-- <a href="#bibtex">BibTeX</a> | -->
    
</p>

<p align="center">
  <span style="background-color: white; padding: 20px; display: inline-block;">
    <img src="assets/corebt_figure.png" width="900">
  </span>
</p>

Juampablo E. Heras Rivera†, Daniel K Low†, Wen-wai Yim, Jacob Ruzevick, Xavier Xiong, Mehmet Kurt*, Asma Ben Abacha* 

† Equal contribution, * Shared last authorship



<div align="center">
<table>
<tr>
<td>


**[KurtLab, University of Washington](https://www.kurtlab.com/)** <br/>
**[Microsoft Health AI, Microsoft](https://www.microsoft.com/en-us/research/lab/microsoft-health-futures/)**

</td>
<!-- <td width="200"></td> spacer column -->
<td align="right">
  <img src="assets/affiliations.png" width="220" alt="BTReport affiliations">
</td>
</tr>
</table>
</div>

![-----------------------------------------------------](assets/purpleline.png)

## Evaluation

To evaluate model predictions, we use:

```bash
python3 eval/evaluate_predictions.py \
  --submission_csv example_submission.csv \
  --reference_csv eval/corebt_sharedtest_groundtruth_alltasks_trainval.csv \
  --task [all, level1, who, lgghgg] \
  --run_id my_run \
  --output_json results.json
```



The submission file provided via `--submission_csv` must be a CSV with one row per subject, see `eval/example_submission.csv` for an example.

| Column | Type | Description | Expected Range | Example |
|------|------|-------------|---------------|--------|
| `subject_id` | string | Unique subject identifier | Any valid subject ID present in the reference CSV | `U0027924` |
| `level1_pred` | integer | Predicted Level-1 tumor class label | 0–3 | `3` |
| `lgghgg_pred` | integer | Predicted LGG/HGG label | 0–1 | `0` |
| `who_grade_pred` | integer | Predicted WHO grade label | 0–2 | `2` |



If `--output_json` is provided, results are saved in a structured JSON file:

```json
{
  "runs": {
    "my_run": {
      "tasks": {
        "level1": {...},
        "lgghgg": {...},
        "who": {...}
      }
    }
  }
}
```


![-----------------------------------------------------](assets/purpleline.png)





<h2 align="center">Repo Structure</h2>

<table align="center">
<tr>
<td><strong><a href="./CLAM/">CLAM/</a></strong></td>
<td>
Data preprocessing and whole-slide tiling utilities based on CLAM<sup>[1]</sup>. 
Includes custom artifact removal using HSV color-based segmentation and tiling pipelines for WSI patch extraction.
</td>
</tr>

<tr>
<td><strong><a href="./gigapath/">gigapath/</a></strong></td>
<td>
Whole-slide histopathology embedding pipeline using the Prov-GigaPath foundation model<sup>[2]</sup>.
Uses tiles generated from CLAM to compute slide-level embeddings.
</td>
</tr>

<tr>
<td><strong><a href="./NeuroVFM/">NeuroVFM/</a></strong></td>
<td>
MRI foundation model framework for multi-sequence brain MRI embedding<sup>[3]</sup>.
Produces subject-level embeddings from T1, T1c, T2, and FLAIR sequences.
</td>
</tr>

<tr>
<td><strong><a href="./dataset_utils/">dataset_utils/</a></strong></td>
<td>
Dataset construction, subject matching, metadata processing, and split generation utilities.
</td>
</tr>

<tr>
<td><strong><a href="./experiments/">experiments/</a></strong></td>
<td>
Scripts for multimodal embedding fusion and downstream tumor typing experiments, including modality ablation studies and evaluation pipelines.
</td>
</tr>
</table>

<br>

<sub>
[1] Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021).  <br>
[2] Xu, Hanwen, et al. "A whole-slide foundation model for digital pathology from real-world data." Nature 630.8015 (2024): 181-188.   <br>
[3] Kondepudi, Akhil, et al. "Health system learning achieves generalist neuroimaging models." arXiv preprint arXiv:2511.18640 (2025).
</sub>

