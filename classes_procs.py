import os
import re
import pandas as pd
from logstuff import FriendlyLog, debug, info, error, warning, critical, not_works_properly
from functools import wraps
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as mape
from settings import SETTINGS
import json
import joblib


class Xl:
    @debug
    def __init__(self, excel_path: str):
        """_summary_

        Args:
            excel_path (str): path to well pasport (.xlsx or .xls)
        """
        self.excel_path = excel_path
        self._fromfile = excel_path
        self.xl = pd.ExcelFile(excel_path)
        self.df = self._parse()
        del self.xl  # to serialize as pck
        self.val_idx = self.get_loc()  # search document entry row

    @debug
    def get_loc(self) -> int:
        """Get the well construction passport entry row

        Raises:
            Exception: raises exception if no entry row was found

        Returns:
            int: index of entry row from which well data can be extracted
        """
        df = self.df.iloc[:5, 7:11]
        for idx, col in enumerate(df.columns):
            vals = df[col].to_list()
            if vals[0] in ['+', 'Не справочное', 'Справочное', '-']:
                continue

            if vals[0] == vals[0]:
                self.wname = vals[0]
                return 7 + idx
        raise Exception

    @debug
    def _parse(self, sheet: str = 'Основная информация') -> pd.DataFrame:
        """Get the well construction passport, it may be conatined in the different excel sheets as listed below

        Args:
            sheet (str, optional): parse sheet with well construction data. Defaults to 'Основная информация'.

        Returns:
            pd.DataFrame: well construction passport 
        """
        try:
            return self.xl.parse(sheet)
        except ValueError:
            try:
                return self.xl.parse('Основная информация-1 ствол')

            except ValueError:
                return self.xl.parse('Основная информация-2 ствол')

    @debug
    def get_ngdu(self) -> str:
        """Get name of field operating company, function looks for a field 'НГДУ' at sheet parsed in _parse() method

        Returns:
            str: _description_
        """
        df = self.df.iloc[:10, :]
        return df[df.iloc[:, 0] == 'НГДУ'].iloc[0, self.val_idx].strip()

    @debug
    def get_field(self) -> str:
        """Get oilfield name, looking for a field 'Месторождение'

        Returns:
            str: field name
        """
        df = self.df.iloc[:10, :]
        return df[df.iloc[:, 0] == 'Месторождение'].iloc[0, self.val_idx].strip()

    @debug
    def get_conductor_depth(self) -> str:
        """get depth of conductor casing end, looking for a field 'Глубина спуска кондуктора'

        Returns:
            str: 
        """
        df = self.df.iloc[:, [0, self.val_idx]]
        return df[df.iloc[:, 0] == 'Глубина спуска кондуктора'].iloc[0, 1]

    @debug
    def get_expl_depth(self) -> str:
        """get production casing end at field 'Глубина спуска эксплуатационной колонны'

        Returns:
            str: _description_
        """
        df = self.df.iloc[:, [0, self.val_idx]]
        return df[df.iloc[:, 0] == 'Глубина спуска эксплуатационной колонны'].iloc[0, 1]


class Well:
    @debug
    def __init__(self, name: str, path: str):
        """Well object, holds specific well data

        Args:
            name (str): well name
            path (str): path to passport (excelfile .xlsx or .xls)
        """
        self.name = name
        self.xl = Xl(path)
        self._fromfile = self.xl._fromfile  # debug
        self.ngdu = self.xl.get_ngdu()
        self.cond_depth = self.xl.get_conductor_depth()
        self.expl_depth = self.xl.get_expl_depth()
        self.field = self.xl.get_field()
        self._success_creation()  # debug

    @info
    def _success_creation(self): return self


class Constr:

    def __init__(self, cond_depth: float, expl_depth: float):
        """Object to store end of casings of interest

        Args:
            cond_depth (float): depth of conductor casing
            expl_depth (float): depth of proudction casing
        """
        if cond_depth == cond_depth:
            self.cd = cond_depth
        else:
            self.cd = -1
        if expl_depth == expl_depth:
            self.ed = expl_depth
        else:
            self.ed = -1


class Model:
    def to_pck(self): ...


class DummyModel:

    def __init__(self, data: 'DataHolder'):
        """Object to store welldata and make prediction

        Args:
            data (DataHolder): Data collected from well passports
        """
        self.data = data
        self.wnames = []
        self.ngdu = []
        self.cond_depth = []
        self.expl_depth = []
        for well in data.wells:
            self.wnames.append(well.name)
            self.ngdu.append(well.ngdu)
            self.cond_depth.append(well.cond_depth)
            self.expl_depth.append(well.expl_depth)
        self.df = pd.DataFrame({'wname': self.wnames, 'ngdu': self.ngdu,
                               'cond_depth': self.cond_depth, 'expl_depth': self.expl_depth})
        self.ngdu = set(self.ngdu)
        self.fullstats = {}
        self._success_creation()  # debug

    @debug
    def check_ngdu(self, ngdu_name: str) -> bool:
        """Ensure that ngdu of interest exists in population

        Args:
            ngdu_name (str): name of oilfield operating company

        Returns:
            bool: flag indicates ngdu exist in population
        """
        if ngdu_name in self.ngdu:
            return True
        else:
            return False

    @info
    def _success_creation(self): return self
    def __repr__(self): return f'{self.__class__} of {self.df.shape}'

    @debug
    def predict(self, ngdu_name: str) -> Constr:
        """Make prediction of well construction of proposed well at some specific ngdu

        Args:
            ngdu_name (str): oil field opearting company name

        Returns:
            Constr: well construction endpoints
        """
        df = self.df
        df_ngdu = df[df['ngdu'] == ngdu_name].copy()
        cond_depth = np.nanmean(df_ngdu.cond_depth) + \
            np.nanstd(df_ngdu.cond_depth)*np.random.random(1)/10
        expl_depth = np.nanmean(df_ngdu.expl_depth) + \
            np.nanstd(df_ngdu.expl_depth)*np.random.random(1)/10
        return Constr(cond_depth[0], expl_depth[0])

    @debug
    def _make_statistics(self):
        """Make and outputs statistics of model prediction abilities.
        See SETTING praph and json path
        """
        df = self.df
        uniq_ngdu = df.ngdu.unique()
        for ngdu in uniq_ngdu:
            df_ngdu = df[df.ngdu == ngdu]

            predicted_ed = []
            predicted_cd = []
            true_ed = []
            true_cd = []
            stats = {}
            for idx, row in df_ngdu.iterrows():
                wname = row.wname
                cd_true = row.cond_depth
                ed_true = row.expl_depth
                contsr_predicted = self.predict(ngdu)

                if cd_true != cd_true or ed_true != ed_true or contsr_predicted.ed != contsr_predicted.ed or contsr_predicted.cd != contsr_predicted.cd:
                    continue

                true_cd.append(cd_true)
                true_ed.append(ed_true)
                well_stats = {}
                well_stats['True CD'] = cd_true
                well_stats['True ED'] = ed_true

                predicted_cd.append(contsr_predicted.cd)
                predicted_ed.append(contsr_predicted.ed)

                well_stats['Predicted CD'] = contsr_predicted.cd
                well_stats['Predicted ED'] = contsr_predicted.ed

                fig, ax = plt.subplots(figsize=(16/1.3, 9/1.3))
                ax.set_title(f'{wname}_{ngdu}')
                ax.set_ylabel('MD, м')
                ax.grid()

                cds = [contsr_predicted.cd, contsr_predicted.ed]
                eds = [cd_true, ed_true]
                X_axis = np.arange(2)
                ax.bar(X_axis - 0.2, cds, 0.4,
                       label='predicted', align='center')
                ax.bar(X_axis + 0.2, eds, 0.4, label='True', align='center')
                ax.set_xticks(
                    X_axis, ['Кондуктор', 'Эксплуатационная колонна'], rotation=0)
                ax.legend()
                fig.savefig(os.path.join(SETTINGS.graph_out,
                            f'{wname}_{ngdu}.{SETTINGS.imgfileext}'))
                plt.close()
                stats[wname] = well_stats

            true_cd = np.array(true_cd)
            true_ed = np.array(true_ed)
            pred_cd = np.array(predicted_cd)
            pred_ed = np.array(predicted_ed)

            stats['True ED mean'] = np.mean(true_ed)
            stats['PREDICTED ED mean'] = np.mean(pred_ed)

            stats['True CD mean'] = np.mean(true_cd)
            stats['PREDICTED CD mean'] = np.mean(pred_cd)

            stats['MAPE ED'] = mape(true_ed, pred_ed)
            stats['MAPE CD'] = mape(true_cd, pred_cd)

            self.fullstats[ngdu] = stats

    @info
    def _dump_stats(self):
        for k, v in self.fullstats.items():
            with open(os.path.join(SETTINGS.json_out, f'{k}.json'), 'w', encoding='utf-8') as f:
                json.dump(v, f, indent=4, ensure_ascii=False)

    @info
    def _dump(self, name='dummymodel1.pck'):
        with open(os.path.join(SETTINGS.models, name), 'wb') as f:
            joblib.dump(self, f, compress=1)

    @staticmethod
    @not_works_properly
    def build_from_pck(pck_path:str) -> 'Model':
        """Make Model from pickled data that obtained from _dump method

        Args:
            pck_path (str): path to pickled bytedata

        Returns:
            Model: Model inst
        """
        with open(pck_path, 'rb') as f:
            return joblib.load(f)

    @not_works_properly
    def update(self): ...


def _worker(name: str, path: str) -> Well:
    """produces obj of type Well. Used for multiprocessing purposes

    Args:
        name (str): well name
        path (str): path to excel

    Returns:
        Well: Well obj
    """
    return Well(name, path)


class DataHolder:
    @debug
    def __init__(self, init_folder: str):
        """makes data scrucutre from bunch of well pasports stored in excel files

        Args:
            init_folder (str): folder with excel files 
        """
        self.init_folder = init_folder
        self.cpus = cpu_count()
        if self.cpus > SETTINGS.cpu_limit:
            self.cpus = SETTINGS.cpu_limit

        self.files = []
        self.wells = []
        for folder, nested_folders, files in os.walk(init_folder):
            for file in tqdm(files, desc=f'reading {folder}'):
                if '~$' in file:
                    continue
                if re.match('.*[.]xls.*', file, flags=re.IGNORECASE):
                    wname = file.split('.')[0]
                    self.files.append((wname, os.path.join(folder, file)))
        # self.files = self.files[:10]
        self.wells = self._get_data()

    @not_works_properly
    def add_well(self, well):
        if isinstance(well, Well):
            self.wells.append(well)

    @not_works_properly
    def add_well_from_xl(self, wname, xlpath):
        self.add_well(Well(wname, xlpath))


    @not_works_properly
    def _get_data(self):
        if True:
            with Pool(processes=self.cpus) as p:
                wells = p.starmap(_worker, self.files)
        else:
            wells = list(map(lambda x: _worker(*x), self.files))
        return wells

    def build_model(self):
        return DummyModel(self)


if __name__ == '__main__':
    init_folder = r'src'
    data = DataHolder(init_folder)
    model = DummyModel(data)
    # model._make_statistics()
    # model._dump_stats()
    # model._dump()
