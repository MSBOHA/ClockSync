import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.analysis.clock_freq_vs_temp import analyze_freq_and_temp
from scripts.analysis.clock_allan_variance import plot_allan_variance

# 设置输入文件路径
data_dir = os.path.join('data', 'long_term_base')
log_file = os.path.join(data_dir, 'log0-original.log')
output_dir = os.path.join('results', 'long_term_base')

# 只绘制频率比（无温度文件）
analyze_freq_and_temp(log_file, temp_file=None, output_dir=output_dir)

# 绘制Allan方差（指定采样点数和m范围）
plot_allan_variance(
    log_file,
    output_dir=output_dir,
    column='ns2s(getT1(RAW))',
    label='Allan 方差 (T1)',
    title='long_term_base RAW T1 的 Allan 方差',
    m_start=2,
    m_end=5000,
    m_step=10,
    max_points=50000
)
