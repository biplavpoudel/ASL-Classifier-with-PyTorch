from flask import Flask
import torch

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return str(device)


if __name__ == '__main__':
    app.run(debug=True)
