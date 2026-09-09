import os
from copy import deepcopy
from typing import Union, IO, List
import yaml
from yacs.config import CfgNode as _CfgNode
from dataset.hdfs_io import hopen


_YAML_EXTS = {"", ".yaml", ".yml"}


class CfgNode(_CfgNode):
    """
    Our own extended version of :class:`yacs.config.CfgNode`.
    It support hdfs config.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict=init_dict, key_list=key_list, new_allowed=new_allowed)

    def update_cfg(self,
                   cfg_filename: str,
                   cfg_str: str = None,
                   cfg_list: List = None) -> None:
        """
        更新一个本地或者hdfs上的yaml文件.
        Args:
            cfg_filename: 本地或者hdfs上的yaml文件名
            cfg_str: cfg string to merge into this CfgNode,
                可以根据string来更新config，格式如下: `{config1}={params1};{config2}={params2}`
                用 `;` 来分割多个参数，用 `=` 来赋值
                比如 "TRAINER.TRAIN_BATCH_SIZE=500;TRAINER.END_EPOCH=32"
            cfg_list: cfg list to merge into this CfgNode。
        """

        def __load_base_cfg(cfg: CfgNode):
            if hasattr(cfg, "BASE") and cfg.BASE:
                for base_config_item in cfg.BASE:
                    base_cfg = self._load_cfg(base_config_item)
                    base_cfg.set_new_allowed(True)
                    # 如果不先load再merge，嵌套三层配置的时候，就会出问题
                    __load_base_cfg(base_cfg)
                    self.merge_from_other_cfg(base_cfg)

        cfg = self._load_cfg(cfg_filename)
        self.set_new_allowed(True)

        __load_base_cfg(cfg)
        self.set_new_allowed(True)
        self.merge_from_other_cfg(cfg)

        if cfg_str:
            self.merge_from_str(cfg_str)

        if cfg_list is not None:
            self.merge_from_list(cfg_list)

        self.CONFIG_PATH = cfg_filename

    def _load_cfg(self, cfg_filename: str):
        """
        加载本地或者hdfs的yaml文件.
        Args:
            cfg_filename: 本地或者hdfs上的yaml文件名
        """
        if cfg_filename.startswith("hdfs://"):
            _, file_extension = os.path.splitext(cfg_filename)
            if file_extension in _YAML_EXTS:
                with hopen(cfg_filename, "r") as fp:
                    return self._load_cfg_from_yaml_str_remote(fp.read())
            else:
                raise ValueError("Not support file type, please check !")
        else:
            with open(cfg_filename, "r") as fp:
                return super().load_cfg(fp)

    def _load_cfg_from_yaml_str_remote(self, str_obj: Union[bytes, IO[bytes], str, IO[str]]):
        """
        通过file_obj加载hdfs上的yaml文件.
        Args:
            str_obj: 文件流stream
        """
        cfg_as_dict = yaml.safe_load(str_obj)
        return _CfgNode(cfg_as_dict)

    def merge_from_str(self, cfg: str) -> None:
        """通过cfg字符串修改当前CfgNode

        Args:
            cfg: cfg字符串，使用"="或者","作分隔符
        """
        cfg_list = cfg.replace('=', ';').split(';')
        for i in range(len(cfg_list)):
            if cfg_list[i].startswith('[') and cfg_list[i].endswith(']'):
                cfg_list[i] = eval(cfg_list[i])
        self.merge_from_list(cfg_list)

    def freeze(self) -> None:
        """
        将当前CfgNode和所有的subCfgNode变成不可以更改的.
        """
        super().freeze()

    def defrost(self) -> None:
        """
        将当前CfgNode和所有的subCfgNode变成可以更改的, 和freeze方法相反.
        """
        super().defrost()

    def clear(self) -> None:
        """
        将当前的CfgNode清空
        """
        if self._is_frozen():
            self.defrost()
        self.clear()

    def dump(self, results_dir: str = "./", config_name: str = "test.yaml") -> None:
        """保存更新后的config到yaml文件中

        Args:
            results_dir (str): 需要保存的yaml文件路径
            config_name (str): 需要保存的yaml文件名
        """
        if not config_name.endswith(".yaml"):
            config_name = config_name + ".yaml"
        if "CONFIG_PATH" in self:
            self.pop("CONFIG_PATH")
        if "MOUDEL_FILE" in self:
            self.pop("MOUDEL_FILE")
        with open(os.path.join(results_dir, f'{config_name}'), 'w') as fw:
            super().dump(stream=fw)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(
                "ERROR: {} not in cfg, please check !".format(name))

    def get(self, names, default=None):
        try:
            names = names.split('.')
            value = self
            for name in names:
                value = value[name]
            return value
        except Exception as e:
            return default

    def _is_frozen(self):
        """ 返回当前cfg是否可以frozen """
        return super().__dict__[CfgNode.IMMUTABLE]


_C = CfgNode()
cfg = _C

# Support multi base config
_C.BASE = []

# ------------------------------------------------------------------------------------- #
# Common options in trainer
# ------------------------------------------------------------------------------------- #
_C.TRAINER = CfgNode()  # type: ignore
_C.TRAINER.RNG_SEED = 12345

# ------------------------------------------------------------------------------------- #
# Common options in accelerators
# ------------------------------------------------------------------------------------- #
_C.ACCELERATOR = CfgNode()  # type: ignore
_C.ACCELERATOR.ACCELERATOR = "ApexDDP"

# ------------------------------------------------------------------------------------- #
# Common network options
# ------------------------------------------------------------------------------------- #
_C.NETWORK = CfgNode()  # type: ignore


default_cfg = deepcopy(_C)


def reset_cfg():
    """ Get default config
    """
    if default_cfg:
        return deepcopy(default_cfg)
