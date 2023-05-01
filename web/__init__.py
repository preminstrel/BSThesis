import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_bootstrap import Bootstrap4
from models.build import build_HPS_model, build_single_task_model
import torch
from data.dataset import get_valid_transforms
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import numpy as np
from torchvision.transforms.functional import normalize, resize, to_pil_image
class myargs():
    def __init__(self, data):
        super(myargs, self).__init__()
        self.data = data
        self.lr = 1e-4
        self.mode = 'train'
        self.batches = 200
        self.epochs = 400

def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    Bootstrap4(app)
    app.config.from_mapping(
        # a default secret that should be overridden by instance config
        SECRET_KEY="dev",
        # store the database in the instance folder
        DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route("/", endpoint="index")
    def index():
        return render_template("index.html")
    
    @app.route("/pdf", endpoint="pdf", methods=['GET', 'POST'])
    def get_pdf():
        filename = request.args.get('filename')
        return send_file(filename, as_attachment=True)

    @app.route("/doc", endpoint="doc")
    def doc():
        return render_template("doc.html")
    
    @app.route("/single", endpoint="single")
    def single():
        return render_template("single.html")
    
    @app.route("/single/upload", endpoint="singleupload", methods=['GET', 'POST'])
    def single_upload():
        if request.method == 'POST':
            uploaded_file = request.files['file']
            uploaded_file.save('/home/hssun/thesis/web/static/img/' + uploaded_file.filename)
            disease = request.form.get('disease')
            
            a = myargs(disease)
            model =build_single_task_model(args=a).eval().cuda()
            best_model_path = f"/home/hssun/thesis/archive/checkpoints/{a.data}/model_best.pth"
            checkpoint = torch.load(best_model_path)
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.decoder.load_state_dict(checkpoint['decoder'])
            
            path = '/home/hssun/thesis/web/static/img/' + uploaded_file.filename
            transform = get_valid_transforms(224)
            img = Image.open(path).convert('RGB')
            img = transform(img).cuda().unsqueeze(0)
            
            img_url = url_for('static', filename='img/' + uploaded_file.filename)
            raw, result = model.inference(img)
            #print(model)
            cam_extractor = SmoothGradCAMpp(model)
            #print(cam_extractor)
            raw = model(img)
            # cam_extractor._hooks_enabled = True
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(raw.squeeze(0).argmax().item(), raw)

            # Resize the CAM and overlay it
            print(path)
            img = np.asarray(Image.open(path).convert('RGB'))
            #print(activation_map[0].squeeze(0))
            cam = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            plt.imshow(cam)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.savefig('/home/hssun/thesis/web/static/img/' + 'out.png', bbox_inches='tight')
            print('saved')
            vis_url = url_for('static', filename='img/' + 'out.png')
        
        return render_template("result_single.html", disease=disease, img_url=img_url, raw=raw, result=result, vis_url=vis_url)

        return {
            "message": "File successfully uploaded",
            "disease": disease,
            "path": '/home/hssun/thesis/web/upload/' + uploaded_file.filename,
            "raw": raw,
            "result": result
        }
    
    @app.route("/multi", endpoint="multi")
    def multi():
        return render_template("multi.html")
    
    @app.route("/multi/upload", endpoint="multiupload", methods=['GET', 'POST'])
    def multi_upload():
        if request.method == 'POST':
            uploaded_file = request.files['file']
            uploaded_file.save('/home/hssun/thesis/web/static/img/' + uploaded_file.filename)
            
            a = myargs("TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+")
            model =build_HPS_model(args=a).cuda()
            best_model_path = "/home/hssun/thesis/archive/checkpoints/HPS/model_best.pth"
            checkpoint = torch.load(best_model_path)
            model.encoder.load_state_dict(checkpoint['encoder'])
            for name, layer in model.named_children():
                if name == "encoder":
                    continue
                model.decoder[name].load_state_dict(checkpoint[name])
            
            path = '/home/hssun/thesis/web/static/img/' + uploaded_file.filename
            transform = get_valid_transforms(224)
            img = Image.open(path).convert('RGB')
            img = transform(img).cuda().unsqueeze(0)
            raw, result = model.inference(img)

            img_url = url_for('static', filename='img/' + uploaded_file.filename)

            #print(model)
            cam_extractor = SmoothGradCAMpp(model)
            #print(cam_extractor)
            raw = model(img)
            # cam_extractor._hooks_enabled = True
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(raw.squeeze(0).argmax().item(), raw)

            # Resize the CAM and overlay it
            print(path)
            img = np.asarray(Image.open(path).convert('RGB'))
            #print(activation_map[0].squeeze(0))
            cam = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            plt.imshow(cam)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.savefig('/home/hssun/thesis/web/static/img/' + 'out.png', bbox_inches='tight')
            print('saved')
            vis_url = url_for('static', filename='img/' + 'out.png')
        
        return render_template("result_multi.html", img_url=img_url, raw=raw, result=result, vis_url=vis_url)
        return {
            "message": "File successfully uploaded",
            "path": '/home/hssun/thesis/web/upload/' + uploaded_file.filename,
            "raw": raw,
            "result": result
        }

    return app