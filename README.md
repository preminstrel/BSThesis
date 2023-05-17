# Bachelor Thesis at SEU
Supervisor: Yi Zhou

![](web/static/img/Thesis.drawio.png)

## Get Started
```bash
conda create -n fundusMTL python=3.9
pip install -r requirements.txt
```

You can find many short scripts in the `script/train.sh` and `script/eval.sh`.
```bash
# train a single-task model baseline on dataset APTOS
python idea.py --data "APTOS" --use_wandb --project APTOS

# train a Hard-Parameters-Sharing model baseline
python idea.py --data "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+" --project HPS --method HPS --multi_task --epochs 200 --num_workers 2 --batch_size 128 --use_wandb
```

The checkpoints will be saved to `archive/checkpoints/` automatically. If you want to modify the frequency or other settings, please refer `utils/parser.py`.

Using Flask to build a web server.
```bash
flask --app web --debug run --port 8500 --host 0.0.0.0
```

![](web/static/img/single.png)

### File Tree
```bash
.
├── LICENSE
├── README.md
├── config_task.py
├── data
│   ├── dataset.py
│   ├── preprocess.py
│   └── sampler.py
├── engine
│   ├── adapter.py
│   ├── adapter_train.py
│   ├── eval.py
│   └── train.py
├── idea.ipynb
├── idea.py
├── main.py
├── moco_train.py
├── models
│   ├── adapter.py
│   ├── build.py
│   ├── discriminator.py
│   ├── encoder_decoder.py
│   ├── loss.py
│   ├── moco.py
│   ├── resnet.py
│   ├── resnet_ca.py
│   ├── resnet_with_adapter.py
│   ├── unet.py
│   └── weighting.py
├── requirements.txt
├── script
│   ├── eval.sh
│   └── train.sh
├── seg_domain_adaptation.py
├── test.ipynb
├── utils
│   ├── data.py
│   ├── image.py
│   ├── info.py
│   ├── metrics.py
│   ├── model.py
│   ├── parser.py
│   └── tsne.py
└── web
    ├── __init__.py
    ├── db.py
    ├── static
    │   ├── css
    │   │   └── index.css
    │   ├── img
    │   │   ├── Thesis.drawio.png
    │   │   ├── index-multi.png
    │   │   ...
    │   └── pdf
    │       ├── proposal.pdf
    │       ...
    └── templates
        ├── doc.html
        ├── footer.html
        ├── index.html
        ├── multi.html
        ├── navi.html
        ├── result_multi.html
        ├── result_single.html
        └── single.html
```
