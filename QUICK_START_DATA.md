# 快速开始使用 akshare 下载数据

## ✅ 已创建的文件

现在您的项目中包含以下数据下载相关文件：

1. **`src/download_csmar_data.py`** - 主要的数据下载脚本
   - 对应 CSMAR 数据库表结构
   - 下载：市值、盈利、ROE、账面价值
   
2. **`src/download_data_akshare.py`** - 通用数据下载器
   - 更灵活的数据下载类
   - 支持批量下载

3. **`DATA_DOWNLOAD.md`** - 详细使用文档

## 🚀 快速使用

### 步骤 1：安装依赖

```bash
cd quant-project
pip install -r requirements.txt
```

### 步骤 2：下载数据

```bash
cd src
python download_csmar_data.py
```

这会下载前 100 只股票的数据作为示例。

### 步骤 3：查看下载的数据

下载完成后，数据会保存在：

```
data/
├── raw/
│   ├── fi_t2.csv           # 财务指标（ROE、盈利等）
│   ├── fs_comins.csv       # 利润表
│   └── fs_combas.csv       # 资产负债表
└── processed/
    └── factor_data.csv     # 处理后的因子数据
```

## 📊 数据说明

### 下载的因子数据包括：

| 因子类型 | 字段名 | 来源表 | 说明 |
|---------|--------|--------|------|
| **市值** | MarketValue | TRD_Mnth | 总市值（含非流通股） |
| **盈利** | NetProfit_Deducted | FI_T2 | 扣除非经常性损益后的净利润（推荐） |
| **盈利** | NetProfit | FS_Comins | 净利润（备选） |
| **ROE** | ROE | FI_T2 | 加权平均净资产收益率 |
| **账面价值** | Equity | FS_Combas | 股东权益/所有者权益 |
| **B/M** | BM | 计算 | 账面市值比（需合并市值） |

## 💡 自定义下载

### 修改下载股票数量

编辑 `download_csmar_data.py`：

```python
# 修改这行（默认 100）
batch_download_all(max_stocks=500)  # 下载 500 只股票
```

### 下载特定股票

```python
from download_csmar_data import download_financial_data

# 下载贵州茅台（600519）的财务数据
fi_t2, fs_comins, fs_combas = download_financial_data('600519')
```

## ⚠️ 注意事项

1. **下载时间**
   - 每只股票约需 2-5 秒
   - 100 只股票约需 3-10 分钟
   - 建议分批下载

2. **数据源**
   - 数据来自东方财富、新浪财经
   - 非官方 CSMAR 数据
   - 学术使用建议交叉验证

3. **网络要求**
   - 需要稳定的网络连接
   - 如遇失败会自动跳过并继续

## 🔍 验证数据

下载完成后，可以用 pandas 查看：

```python
import pandas as pd

# 查看财务指标数据
df = pd.read_csv('../data/raw/fi_t2.csv')
print(df.head())
print(df.columns)

# 查看因子数据
factor_df = pd.read_csv('../data/processed/factor_data.csv')
print(factor_df.head())
```

## 📝 下一步

数据下载完成后，您可以：

1. 使用现有的量化策略框架进行回测
2. 运行因子分析
3. 构建投资组合

现在 GitHub 仓库中应该有完整的文件了！访问 https://github.com/Tantan-ava/quant-project 查看。
