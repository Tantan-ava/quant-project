# 数据下载指南

## 使用 akshare 下载 CSMAR 类型数据

本项目的数据下载脚本使用 akshare 从公开数据源获取类似 CSMAR 数据库的量化因子数据。

### 数据表对应关系

| CSMAR 表名 | 描述 | akshare 替代数据源 |
|-----------|------|-------------------|
| TRD_Mnth | 月度交易数据（市值、收益率） | `stock_zh_a_daily` + `stock_zh_a_spot_em` |
| FI_T2 | 财务指标（ROE、盈利） | `stock_financial_analysis_indicator_em` |
| FS_Comins | 利润表（净利润） | `stock_financial_report_sina` |
| FS_Combas | 资产负债表（股东权益） | `stock_financial_report_sina` |

### 快速开始

#### 1. 安装依赖

```bash
pip install akshare pandas numpy openpyxl
```

#### 2. 下载数据

```bash
cd src
python download_csmar_data.py
```

#### 3. 数据说明

**下载的数据包括：**

- **市值数据**（来自 TRD_Mnth 类似表）
  - 总市值（包含非流通股）
  - 流通市值
  - 月度收益率

- **盈利数据**（来自 FI_T2 或 FS_Comins）
  - 扣除非经常性损益后的净利润（推荐）
  - 净利润

- **ROE 数据**（来自 FI_T2）
  - 加权平均净资产收益率

- **账面价值**（来自 FS_Combas）
  - 股东权益/所有者权益
  - 每股净资产

### 主要函数说明

#### `download_stock_list()`
获取所有 A 股股票代码列表。

#### `download_monthly_returns(stock_code)`
下载个股月度收益率数据，类似 TRD_Mnth 表。

#### `download_financial_data(stock_code)`
下载个股财务数据，返回三个 DataFrame：
- FI_T2：财务指标
- FS_Comins：利润表
- FS_Combas：资产负债表

#### `process_factors(fi_t2, fs_comins, fs_combas)`
处理因子数据，提取：
- ROE（净资产收益率）
- NetProfit_Deducted（扣非净利润）
- Equity（股东权益）
- BM（账面市值比，需要合并市值数据）

### 批量下载

```python
# 下载前 500 只股票的数据
batch_download_all(max_stocks=500)
```

### 数据保存位置

- **原始数据**: `data/raw/`
  - `fi_t2.csv` - 财务指标
  - `fs_comins.csv` - 利润表
  - `fs_combas.csv` - 资产负债表

- **处理后数据**: `data/processed/`
  - `factor_data.csv` - 合并后的因子数据

### 注意事项

1. **数据源限制**
   - akshare 数据来自公开渠道（东方财富、新浪等）
   - 非官方 CSMAR 数据，可能存在差异
   - 建议用于研究，实盘需交叉验证

2. **下载速度**
   - 批量下载时建议控制数量
   - 每 50 只股票会自动保存临时文件
   - 网络不稳定时可减少单次下载量

3. **数据更新**
   - 财务数据按报告期更新
   - 建议每月更新一次
   - 市值数据可每日更新

### 常见问题

**Q: 为什么有些股票下载失败？**
A: 可能是股票代码格式问题，确保使用 6 位数字代码（如 600519）。

**Q: 数据与 CSMAR 有差异？**
A: akshare 使用公开数据源，计算口径可能不同，建议交叉验证。

**Q: 如何获取历史市值数据？**
A: 目前 akshare 主要提供实时市值，历史市值需要通过日行情数据估算。

### 相关资源

- [akshare 官方文档](https://akshare.akfamily.xyz/)
- [东方财富财务数据接口](https://akshare.akfamily.xyz/data/stock/stock.html)
- [新浪财经财务数据](https://akshare.akfamily.xyz/data/stock/stock.html)
