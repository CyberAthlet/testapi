if __name__ =='__main__':
    """This module should be used when there is a need to
    update model pickle source. Normally dont use that
    """
    from logstuff import FriendlyLog, debug, info, error, warning, critical
    from classes_procs import Model, DummyModel, DataHolder, Well, Xl
    from settings import SETTINGS
    from fastapi import FastAPI

    # init_folder = r'src'
    # data = DataHolder(init_folder)
    # model = DummyModel(data)
    # model = DummyModel.build_from_pck('out\models\dummymodel1.pck')
    # print(model)
    import openpyxl
    import pandas
    import gunicorn 
    print(gunicorn.__version__)
    # model.plot_statistics()
    # model.dump_stats()
    # model.dump()
