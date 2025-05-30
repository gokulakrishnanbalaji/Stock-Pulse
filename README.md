# Installation instructions

create venv, dont install the dependencies in your global python interpreter

only use this if you're using conda
```bash
conda create --name stockpulse  python=3.9 -y
conda activate stockpulse
```

Install required requirements in python env
```bash
pip install -r requirements.txt
```

Initialise dvc
```bash
dvc init
```

Make the sh file executable
```bash
chmod 777 run_pipeline.sh
```

Run the pipeline
``` bash
./run_pipeline.sh
```

Opne the mlflow console at `http://127.0.0.1:5000/`
``` bash
mlflow ui
```

open  `index.html` file inside `ui` folder in a brower to see UI

Perform unit testing
```bash
pytest src/test.py -v
```

Stop the backend fast api server (if you wish to stop running the app)
```bash
pkill -f "python src/backend.py"
```

use the command inside `cron.txt ` to set it up as a cron job.

# High level design

<img width="1072" alt="image" src="https://github.com/user-attachments/assets/1c50a882-5fa2-472e-9adc-d0ec1383cc97" />

