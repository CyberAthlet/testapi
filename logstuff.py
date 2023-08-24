import logging
from functools import  wraps
import inspect
import os
import traceback
from settings import SETTINGS

class FriendlyLog:
    _reg = []
    loglevel = 'DEBUG'
    fpath = os.path.join(os.path.dirname(__file__), SETTINGS.logpath)
    log_format = SETTINGS.logfmt
    file_handler = logging.FileHandler(fpath, mode = 'a', encoding= 'utf-8')
    formatter = logging.Formatter(log_format, datefmt=SETTINGS.datefmt)
    file_handler.setFormatter(formatter)

    with open(fpath, 'w') as f: ...

    @classmethod
    def set_level(cls, level):
        cls.loglevel = level
        for item in cls._reg:
            item._update()

    def __init__(self,msglevel = 'DEBUG'):
        if len(self._reg) >=5:
            raise Exception('You dont need more logging levels')
        FriendlyLog._reg.append(self)
        self.level = msglevel
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(self.loglevel)    
        self._logger.addHandler(self.file_handler)
        logging_methods = {
        'debug': self._logger.debug,
        'info': self._logger.info,
        'warning': self._logger.warning,
        'error': self._logger.error, 
        'critical': self._logger.critical
        }
        
        self.log = logging_methods.get(msglevel.lower(), self._logger.debug)

    def _update(self):
        self._logger.setLevel(self.loglevel)

    def __call__(self, *args, **kwargs):
        skip = kwargs.get('skip', False)

        def outer_wrapee(f):
            def wrapee(*args, **kwargs):
                frm = inspect.stack()[1]
                mod = inspect.getmodule(frm[0])
                msg = f'CALL {f.__name__} from {f.__module__} in {mod}'
                debugmsg = f'{args=} {kwargs=} defaults = {f.__defaults__ }'
                try:
                    val = f(*args, **kwargs)
                except Exception as ex:
                    msg = 'FAIL ' + msg + ' ' + debugmsg
                    tb = traceback.format_exc()
                    self._logger.error(f'{msg}\n{tb}')
                    if not skip:
                        raise ex
                    else:
                        return None
                else:
                    msg = 'SUCCES ' + msg 
                    if self.level =='debug':
                        self.log(msg + ' ' + debugmsg + f' {val = }')
                    else:                
                        self.log(msg + f' {val = }')
                    return val
            return wrapee
    
        if args:
            if callable(args[0]):
                return outer_wrapee(args[0])

        return outer_wrapee

debug = FriendlyLog(msglevel = 'debug')
info = FriendlyLog(msglevel ='info')
error = FriendlyLog(msglevel ='error')
warning = FriendlyLog(msglevel ='warning')
critical = FriendlyLog(msglevel ='critical')

def not_works_properly(func):

    @wraps(func)
    def bad_func(*args, **kwargs):
        warning.log(f'FUNC {func.__name__} from {func.__module__} called which can bad behave')   
        return func(*args, **kwargs)     
    return bad_func
