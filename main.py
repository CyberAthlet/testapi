#import uvicorn
from logstuff import FriendlyLog, debug, info, error, warning, critical
from classes_procs import Model, DummyModel, DataHolder, Well, Xl
from settings import SETTINGS
from fastapi import FastAPI
from fastapi import Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import BaseModel, Field
from fastapi.responses import PlainTextResponse


FriendlyLog.set_level('INFO')

model = DummyModel.build_from_pck(r'out/models/dummymodel1.pck')


class Prediction(BaseModel):
    class Config:
        populate_by_name = True

    ngdu: str = Field(None, alias='НГДУ')
    cd: float = Field(None, alias='Глубина кондуктора')
    ed: float = Field(None, alias='Глубина эксплуатационной колонны')
    cd_mean: float = Field(None, alias='Средняя глубина кондуктора')
    ed_mean: float = Field(
        None, alias='Средняя глубина эксплуатационной колонны')
    MAPE_CD: float = Field(None, alias='MAPE глубины кондуктора')
    MAPE_ED: float = Field(None, alias='MAPE эксплуатационной колонны')



app = FastAPI()

@app.get('/')
def root():
    s = [f'/predict/{x}' for x in model.fullstats.keys()]
    msg = {'MESSAGE': s}
    return JSONResponse(content=msg, status_code=200)

@app.get('/predict')
async def predict():
    s = [f'/predict/{x}' for x in model.fullstats.keys()]
    raise HTTPException(status_code=404, detail=s)


@app.get('/predict/{ngdu_name}')
async def get_constr(ngdu_name: str):
    if model.check_ngdu(ngdu_name):
        constr = model.predict(ngdu_name)
        return JSONResponse(Prediction(**{'ngdu': ngdu_name,
                                          'cd': constr.cd,
                                          'ed': constr.ed,
                                          'cd_mean': model.fullstats[ngdu_name]['True CD mean'],
                                          'ed_mean': model.fullstats[ngdu_name]['True ED mean'],
                                          'MAPE_CD': model.fullstats[ngdu_name]['MAPE CD'],
                                          'MAPE_ED': model.fullstats[ngdu_name]['MAPE ED'],
                                          }).model_dump(by_alias=True), status_code=200)
    else:
        d = {'options':list(model.fullstats.keys())}
        raise HTTPException(status_code=404, detail=d)

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="info")
