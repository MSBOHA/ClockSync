"""
配置模块
管理分析参数和设置
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class AnalysisConfig:
    """分析配置类"""
    # 数据过滤参数
    lower_percent: float = 0.01
    upper_percent: float = 0.99
    
    # 拟合参数
    fit_method: str = 'svr'  # 'svr' 或 'lsq'
    fit_lower_percent: float = 0.01
    fit_upper_percent: float = 0.3
    piecewise_fit: bool = False
    piece_num: int = 3
    
    # GMM参数
    enable_gmm: bool = True
    gmm_n_components_range: Tuple[int, int] = (1, 6)
    gmm_log_transform: bool = True
    
    # 绘图参数
    plot_real_time: bool = False
    plot_raw_delay: bool = False
    show_plots: bool = False
    
    # 输出参数
    output_dir: str = 'results/refactored_analysis'
    save_detailed_results: bool = True
    
    # 批处理参数
    batch_mode: bool = False
    input_files: List[str] = field(default_factory=list)
    
    @classmethod
    def from_args(cls, args):
        """从命令行参数创建配置"""
        return cls(
            lower_percent=getattr(args, 'lower_percent', 0.01),
            upper_percent=getattr(args, 'upper_percent', 0.99),
            fit_method=getattr(args, 'fit_method', 'svr'),
            fit_lower_percent=getattr(args, 'fit_lower_percent', 0.01),
            fit_upper_percent=getattr(args, 'fit_upper_percent', 0.3),
            piecewise_fit=getattr(args, 'piecewise_fit', False),
            piece_num=getattr(args, 'piece_num', 3),
            enable_gmm=getattr(args, 'enable_gmm', True),
            gmm_n_components_range=(
                getattr(args, 'gmm_min_components', 1),
                getattr(args, 'gmm_max_components', 6)
            ),
            gmm_log_transform=getattr(args, 'gmm_log_transform', True),
            plot_real_time=getattr(args, 'plot_real_time', False),
            plot_raw_delay=getattr(args, 'plot_raw_delay', False),
            show_plots=getattr(args, 'show_plots', False),
            output_dir=getattr(args, 'output_dir', 'results/refactored_analysis'),
            save_detailed_results=getattr(args, 'save_detailed_results', True),
            batch_mode=getattr(args, 'batch_mode', False),
            input_files=getattr(args, 'input_files', [])
        )
    
    def validate(self):
        """验证配置参数"""
        if not (0 <= self.lower_percent < self.upper_percent <= 1):
            raise ValueError("分位数参数必须满足: 0 <= lower_percent < upper_percent <= 1")
        
        if self.fit_method not in ['svr', 'lsq']:
            raise ValueError("拟合方法必须是 'svr' 或 'lsq'")
        
        if self.piece_num < 1:
            raise ValueError("分段数必须大于等于1")
        
        if not (0 <= self.fit_lower_percent < self.fit_upper_percent <= 1):
            raise ValueError("拟合分位数参数必须满足: 0 <= fit_lower_percent < fit_upper_percent <= 1")
        
        if not (1 <= self.gmm_n_components_range[0] <= self.gmm_n_components_range[1] <= 10):
            raise ValueError("GMM组件数范围必须在1-10之间")
    
    def get_output_subdir(self, load_level=None):
        """获取特定负载等级的输出子目录"""
        if load_level:
            return f"{self.output_dir}/load_{load_level}"
        return self.output_dir
