import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
})

// ── Interfaces ─────────────────────────────────────────────

export interface PredictRequest {
  gpu_name: string
  model_name: string
  gpu_count: number
  input_tokens: number
  output_tokens: number
  concurrency: number
  max_ttft_ms?: number
  min_throughput_per_user?: number
}

export interface PredictResult {
  predicted_ttft_mean_ms: number | null
  predicted_ttft_p90_ms: number | null
  predicted_throughput_tokens_s: number | null
  max_safe_concurrency: number | null
  recommended_concurrency: number | null
}

export interface PredictResponse {
  source: 'interpolation' | 'model_based' | 'ensemble' | 'unavailable'
  confidence: number
  data_points_used: number
  result: PredictResult
  warnings: string[]
  metadata: Record<string, unknown>
}

export interface CostOptimizeRequest {
  target_concurrency: number
  model_name: string
  input_tokens: number
  output_tokens: number
  max_ttft_ms: number
  min_throughput_per_user: number
  top_k?: number
}

export interface CostRecommendation {
  rank: number
  gpu_name: string
  gpu_count: number
  price_per_hour: number
  max_concurrency: number | null
  utilization_rate: number | null
  cost_per_1m_tokens: number | null
  source: string
  confidence: number
  warnings: string[]
}

export interface CostOptimizeResponse {
  recommendations: CostRecommendation[]
}

export interface DataOptionsItem {
  gpu_name: string
  model_name: string
  gpu_count: number
  input_tokens: number[]
  output_tokens: number[]
}

export interface CoverageItem {
  gpu_name: string
  model_name: string
  gpu_count: number
  data_count: number
  min_input?: number
  max_input?: number
  min_concurrency?: number
  max_concurrency?: number
}

export interface DataCoverageResponse {
  total_rows: number
  items: CoverageItem[]
}

export interface GpuSpec {
  id?: number
  name: string
  vendor?: string
  memory_gb: number
  memory_bandwidth_gbps: number
  tflops_bf16: number
  price_per_hour: number
  notes?: string
}

export interface Model {
  id?: number
  name: string
  parameter_b: number
  model_type?: string
  default_model_path?: string
  notes?: string
}

export interface BenchmarkRun {
  task_id: string
  status: string
  gpu_name: string
  model_name: string
  gpu_count: number
  created_at: string
  finished_at: string | null
  error_message: string | null
}

export interface SweepPoint {
  concurrency: number
  throughput_tokens_s: number | null
  throughput_per_user_tokens_s: number | null
  ttft_mean_ms: number | null
  ttft_s: number | null
  decode_latency_mean_ms: number | null
  source: string | null
  confidence: number | null
  /** 该点的并发数超出数据库实测范围，值为边界截断（非真实外推），不应用于安全并发计算 */
  is_extrapolation?: boolean
}

export interface SweepRequest {
  gpu_name: string
  model_name: string
  gpu_count: number
  input_tokens: number
  output_tokens: number
  max_concurrency: number
}

export interface SubmitBenchmarkRequest {
  api_base_url: string
  api_key: string
  backend_type: 'openai' | 'openai-chat'
  model_name: string
  gpu_name: string
  gpu_count: number
  input_tokens_list: number[]
  output_tokens: number
  concurrency_list: number[]
  /** 每个测试点的轮数。每轮同时发 concurrency 个请求，总请求数 = epochs × concurrency */
  epochs: number
  /** 首 token 延时均值上限（ms），超出后停止更高并发测试，null 表示不限制 */
  max_ttft_ms?: number | null
  /** 单用户吞吐下限（tokens/s），低于则停止更高并发测试，null 表示不限制 */
  min_throughput_per_user?: number | null
  /** 分词器路径或 modelscope/huggingface 模型名，用于精确 token 计数；留空自动尝试 model_name */
  tokenizer_path?: string | null
}

export interface BenchmarkPointResult {
  input_tokens: number
  output_tokens: number
  concurrency: number
  throughput_tokens_s: number | null
  throughput_per_user_tokens_s: number | null
  ttft_mean_ms: number | null
  ttft_p90_ms: number | null
  ttft_p99_ms: number | null
  ttft_max_ms: number | null
  decode_latency_mean_ms: number | null
  decode_latency_p90_ms: number | null
  decode_latency_p99_ms: number | null
  decode_latency_max_ms: number | null
}

export interface BenchmarkDataRecord {
  id: number
  gpu_name: string
  model_name: string
  gpu_count: number
  input_tokens: number
  output_tokens: number
  concurrency: number
  throughput_tokens_s: number | null
  throughput_per_user_tokens_s: number | null
  ttft_mean_ms: number | null
  ttft_p90_ms: number | null
  ttft_p99_ms: number | null
  ttft_max_ms: number | null
  decode_latency_mean_ms: number | null
  decode_latency_p90_ms: number | null
  decode_latency_p99_ms: number | null
  decode_latency_max_ms: number | null
  recorded_at: string | null
}

export interface BenchmarkRecordsResponse {
  total: number
  page: number
  page_size: number
  items: BenchmarkDataRecord[]
}

export interface ImportResponse {
  rows_imported: number
  sheets_processed: number
  skipped: string[]
}

// ── API Functions ─────────────────────────────────────────

export const predict = (req: PredictRequest): Promise<PredictResponse> =>
  api.post<PredictResponse>('/predict', req).then((r) => r.data)

export const predictSweep = (req: SweepRequest): Promise<SweepPoint[]> =>
  api.get<SweepPoint[]>('/predict/sweep', { params: req }).then((r) => r.data)

export const costOptimize = (params: CostOptimizeRequest): Promise<CostOptimizeResponse> =>
  api.get<CostOptimizeResponse>('/cost/optimize', { params }).then((r) => r.data)

export const submitBenchmark = (config: SubmitBenchmarkRequest): Promise<{ task_id: string }> =>
  api.post<{ task_id: string }>('/benchmark/run', config).then((r) => r.data)

export const getBenchmarkStatus = (taskId: string): Promise<BenchmarkRun> =>
  api.get<BenchmarkRun>(`/benchmark/${taskId}/status`).then((r) => r.data)

export const listBenchmarkRuns = (): Promise<BenchmarkRun[]> =>
  api.get<BenchmarkRun[]>('/benchmark/list').then((r) => r.data)

export const getBenchmarkResults = (taskId: string): Promise<BenchmarkPointResult[]> =>
  api.get<BenchmarkPointResult[]>(`/benchmark/${taskId}/results`).then((r) => r.data)

export const cancelBenchmark = (taskId: string): Promise<{ task_id: string; status: string }> =>
  api.post<{ task_id: string; status: string }>(`/benchmark/${taskId}/cancel`).then((r) => r.data)

export const listBenchmarkData = (params: {
  gpu_name?: string; model_name?: string; gpu_count?: number; input_tokens?: number
  page?: number; page_size?: number
}): Promise<BenchmarkRecordsResponse> =>
  api.get<BenchmarkRecordsResponse>('/data/records', { params }).then((r) => r.data)

export const updateBenchmarkData = (
  id: number, body: Partial<Omit<BenchmarkDataRecord, 'id' | 'gpu_name' | 'model_name' | 'gpu_count' | 'input_tokens' | 'output_tokens' | 'concurrency' | 'recorded_at'>>
): Promise<BenchmarkDataRecord> =>
  api.put<BenchmarkDataRecord>(`/data/records/${id}`, body).then((r) => r.data)

export const deleteBenchmarkData = (ids: number[]): Promise<{ deleted: number }> => {
  const params = new URLSearchParams()
  ids.forEach((id) => params.append('ids', String(id)))
  return api.delete<{ deleted: number }>(`/data/records?${params.toString()}`).then((r) => r.data)
}

export const getDataCoverage = (params?: { gpu_name?: string; model_name?: string }): Promise<DataCoverageResponse> =>
  api.get<DataCoverageResponse>('/data/coverage', { params }).then((r) => r.data)

export const getDataOptions = (): Promise<DataOptionsItem[]> =>
  api.get<DataOptionsItem[]>('/data/options').then((r) => r.data)

export const importData = (formData: FormData): Promise<ImportResponse> =>
  api.post<ImportResponse>('/data/import', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)

export const reimportData = (): Promise<ImportResponse> =>
  api.post<ImportResponse>('/data/reimport').then((r) => r.data)

// GPU CRUD
export const listGpus = (): Promise<GpuSpec[]> =>
  api.get<GpuSpec[]>('/gpus').then((r) => r.data)

export const createGpu = (data: Omit<GpuSpec, 'id'>): Promise<GpuSpec> =>
  api.post<GpuSpec>('/gpus', data).then((r) => r.data)

export const updateGpu = (id: number, data: Partial<GpuSpec>): Promise<GpuSpec> =>
  api.put<GpuSpec>(`/gpus/${id}`, data).then((r) => r.data)

export const deleteGpu = (id: number): Promise<void> =>
  api.delete(`/gpus/${id}`).then((r) => r.data)

// Model CRUD
export const listModels = (): Promise<Model[]> =>
  api.get<Model[]>('/models').then((r) => r.data)

export const createModel = (data: Omit<Model, 'id'>): Promise<Model> =>
  api.post<Model>('/models', data).then((r) => r.data)

export const updateModel = (id: number, data: Partial<Model>): Promise<Model> =>
  api.put<Model>(`/models/${id}`, data).then((r) => r.data)

export const deleteModel = (id: number): Promise<void> =>
  api.delete(`/models/${id}`).then((r) => r.data)

export default api
