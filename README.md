# Isotope Peak Deconvolution Service

供高分辨质谱（HRMS）实验室复核重叠同位素峰的**纯后端服务**：Python 3.13 + FastAPI，
无前端。分析员提交一组按质荷比严格递增的峰，服务执行**确定性、穷举式**解卷积，
返回峰簇划分与裁决（`UNIQUE` / `AMBIGUOUS` / `UNRESOLVED`）。

## 问题定义

- 输入：2–36 个峰（`mz` 为正的十进制小数、严格递增；`intensity` 为正整数）、
  允许电荷集合 `charges`（正整数、互不重复）、十进制容差 `tolerance`（≥ 0）。
- 峰簇：2–6 个峰、同一电荷 `z`，相邻质荷比之差与 `1.003355 / z` 的偏差不超过容差
  （内部以 `|Δmz·z − 1.003355| ≤ tolerance·z` 精确判定，无浮点误差）。
- 每个峰至多属于一个峰簇。
- 求解器**完整搜索所有合法峰簇组合**（基于位掩码的精确动态规划，非贪心、
  非"逐峰就近"、非"先选最强候选"），按字典序依次优化：
  1. 最大化已解释总强度；
  2. 最大化已解释峰数；
  3. 最小化峰簇数。
- 裁决：
  - `UNIQUE`：最优组合唯一；
  - `AMBIGUOUS`：三项目标完全相同的最优组合不止一个，响应附带一份不同的
    见证（`second_witness`）；
  - `UNRESOLVED`：不存在任何合法峰簇。
- 非法输入返回 422，错误体给出可定位字段（`error.fields[].loc`），且不产生裁决。

> **安全阀**：搜索始终保持穷举；仅当输入病态（如容差接近同位素间距本身，
> 集合打包搜索空间指数爆炸）导致工作量超过预算时，服务返回 503
> （`SEARCH_SPACE_EXCEEDED`）而非挂起，绝不返回错误裁决。预算可通过环境变量
> `DECONVOLVER_MAX_SEARCH_OPS` 调整（默认 20,000,000 次簇扩展操作）。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/deconvolve` | 解卷积裁决（版本化 JSON 接口） |
| POST | `/api/v1/deconvolve/coeluting` | 同一前体的多电荷共洗脱包络确认 |
| GET | `/health` | 健康检查 |
| GET | `/docs` | OpenAPI 交互文档 |

### 请求示例

```bash
curl -s http://localhost:8000/api/v1/deconvolve \
  -H 'Content-Type: application/json' \
  -d '{
        "peaks": [
          {"mz": "500.000000", "intensity": 1000},
          {"mz": "501.003355", "intensity": 800},
          {"mz": "502.006710", "intensity": 600}
        ],
        "charges": [1],
        "tolerance": "0.0005"
      }'
```

### 响应示例（节选）

```json
{
  "verdict": "UNIQUE",
  "objectives": {"explained_intensity": 2400, "explained_peak_count": 3, "cluster_count": 1},
  "clusters": [
    {"charge": 1, "peak_indices": [0, 1, 2], "explained_intensity": 2400,
     "peaks": [{"index": 0, "mz": "500.000000", "intensity": 1000}, "..."]}
  ],
  "unexplained_peaks": [],
  "second_witness": null,
  "input_summary": {"peak_count": 3, "charges": [1], "tolerance": "0.0005", "isotope_spacing": "1.003355"}
}
```

`clusters` 按（首峰 m/z、电荷、峰下标）规范排序；`mz` 以字符串原样返回以保持十进制精度。

### 错误响应（422）

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input; no deconvolution verdict was produced.",
    "fields": [{"loc": "peaks.1.intensity", "message": "Input should be greater than 0", "type": "greater_than"}]
  }
}
```

## 多电荷共洗脱确认（POST /api/v1/deconvolve/coeluting）

用于确认**同一化合物**在多个电荷态下的同位素包络，避免把各电荷态"分别最优"
的峰簇误合并为同一前体。请求沿用 `peaks`、`charges`、`tolerance`，并新增：

- `required_charges`：应同时出现的 **2–4 个**电荷态；必须为正整数、互不重复，
  且是 `charges` 的子集，否则返回 422（可定位字段，不产生裁决）。
- `mass_tolerance`：中性质量容差（十进制，≥ 0），非法时同样返回 422。

语义：

- 每个指定电荷态**恰好选一个**合法峰簇，簇间互不重叠；
- 每个簇的中性质量估计为 `首峰 m/z × 电荷`，其容差区间
  `[M − mass_tolerance, M + mass_tolerance]` 必须存在**公共交集**
  （等价于 `max(M) − min(M) ≤ 2·mass_tolerance`，边界含等号）；
- 求解器在**满足共同中性质量与指定电荷覆盖的全部候选组合**中直接穷举，
  仍按字典序依次最大化已解释强度、最大化已解释峰数、最小化簇数——
  **不是**先运行普通解卷积再筛选；
- 响应返回规范排序的簇、公共质量区间 `common_mass_interval`
  （`[max(M) − tol, min(M) + tol]`，字符串以保持十进制精度）、
  `UNIQUE` / `AMBIGUOUS` / `UNRESOLVED` 裁决，以及（AMBIGUOUS 时）一份不同的
  见证 `second_witness`（含其自身的公共质量区间）。

### 请求示例

```bash
curl -s http://localhost:8000/api/v1/deconvolve/coeluting \
  -H 'Content-Type: application/json' \
  -d '{
        "peaks": [
          {"mz": "250.000000", "intensity": 400},
          {"mz": "250.5016775", "intensity": 300},
          {"mz": "500.000000", "intensity": 1000},
          {"mz": "501.003355", "intensity": 800}
        ],
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
        "mass_tolerance": "0.5"
      }'
```

### 响应示例（节选）

```json
{
  "verdict": "UNIQUE",
  "objectives": {"explained_intensity": 2500, "explained_peak_count": 4, "cluster_count": 2},
  "clusters": [
    {"charge": 2, "peak_indices": [0, 1], "explained_intensity": 700, "peaks": ["..."]},
    {"charge": 1, "peak_indices": [2, 3], "explained_intensity": 1800, "peaks": ["..."]}
  ],
  "common_mass_interval": {"lower": "499.500000", "upper": "500.500000"},
  "unexplained_peaks": [],
  "second_witness": null,
  "input_summary": {"peak_count": 4, "charges": [1, 2], "required_charges": [1, 2],
                    "tolerance": "0.0001", "mass_tolerance": "0.5", "isotope_spacing": "1.003355"}
}
```

## 快速开始（Docker）

```bash
# 构建并启动 API（宿主机端口默认 8000，可用 API_PORT 覆盖）
docker compose up --build

# 自定义宿主机端口
API_PORT=9000 docker compose up --build

# 一次性运行真实接口验收（verify 服务，依赖 api 健康检查后自动执行）
docker compose run --rm verify
# 或：docker compose --profile verify up --abort-on-container-exit
```

`verify` 服务对运行中的真实 API 执行全部验收场景（UNIQUE / AMBIGUOUS /
UNRESOLVED、字典序目标、容差边界、36 峰全量、非法输入 422，以及共洗脱接口的
共同质量边界、全局取舍、多解见证与兼容回归等），全部通过时退出码为 0。

## 本地开发

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

uvicorn app.main:app --reload --port 8000   # 启动服务
pytest                                       # 单元 / API 测试
python verify/verify_acceptance.py           # 对本机实例跑验收（API_BASE_URL 可覆盖）
```

## 目录结构

```
app/
  main.py     # FastAPI 应用、路由、错误处理
  schemas.py  # 请求/响应模型（Pydantic 校验）
  solver.py   # 穷举式精确求解器（Decimal 精确运算）
tests/        # pytest 单元与接口测试
verify/       # 一次性真实接口验收脚本（compose 的 verify 服务）
Dockerfile
docker-compose.yml
```
