import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

import base64

defaults.device = torch.device('cpu')

export_file_name = 'sec_train.pkl'

MEDICINE = {
    '1': 'Paracetamol',
    '4': 'Cetirizine',
    '7': 'Dompiridone',
    '9': 'Corex',
}

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='static'))

async def setup_learner():
    try:
        learn = load_learner('.\models',export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction, tensor, prob = learn.predict(img)#[0]
    print('prob', prob)

    return JSONResponse({'result': MEDICINE[str(prediction)]})

# @app.route('/analyze', methods=['POST'])
# async def analyze(request):
    
#     #img_data = await request.form()
#     imgdata = base64.b64decode(request['image']).decode("ascii")
#     #img_bytes = await (img_data['file'].read())
#     img_bytes = await (img_data.read())

#     img = open_image(BytesIO(img_bytes))
#     prediction, tensor, prob = learn.predict(img)  # [0]
#     print('prob', prob)

#     return JSONResponse({'result': MEDICINE[str(prediction)]})



if __name__ == '__main__':
    #if 'serve' in sys.argv:
    uvicorn.run(app=app, host='127.0.0.1', port=5000, log_level="info")
    #uvicorn.run(app=app, host='192.168.43.208', port=5000, log_level="info")
