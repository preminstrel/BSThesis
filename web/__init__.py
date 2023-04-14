import os
from flask import Flask, render_template, request, redirect, url_for
from web import db
from flask_bootstrap import Bootstrap4
from models.build import build_HPS_model, build_single_task_model
import torch
from data.dataset import get_valid_transforms
from PIL import Image

class myargs():
    def __init__(self, data):
        super(myargs, self).__init__()
        self.data = data
        self.lr = 1e-4
        self.mode = 'eval'
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
    
    @app.route("/single", endpoint="single")
    def index():
        return render_template("single.html")
    
    @app.route("/single/upload", endpoint="singleupload", methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            uploaded_file = request.files['file']
            uploaded_file.save('/home/hssun/thesis/web/upload/' + uploaded_file.filename)
            disease = request.form.get('disease')
            
            a = myargs(disease)
            model =build_single_task_model(args=a).cuda()
            best_model_path = f"/home/hssun/thesis/archive/checkpoints/{a.data}/model_best.pth"
            checkpoint = torch.load(best_model_path)
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.decoder.load_state_dict(checkpoint['decoder'])
            path = '/home/hssun/thesis/web/upload/' + uploaded_file.filename
            transform = get_valid_transforms(224)
            img = Image.open(path).convert('RGB')
            img = transform(img).cuda().unsqueeze(0)
            raw, result = model.inference(img)
        
        return {
            "message": "File successfully uploaded",
            "disease": disease,
            "path": '/home/hssun/thesis/web/upload/' + uploaded_file.filename,
            "raw": raw,
            "result": result
        }

    
    # register the database commands
    db.init_app(app)

    return app