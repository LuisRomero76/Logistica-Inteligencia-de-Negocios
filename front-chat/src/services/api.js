import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error('API Error:', error.response.data)
    } else if (error.request) {
      console.error('Network Error:', error.message)
    }
    return Promise.reject(error)
  }
)

export default {
  /**
   * POST /predict - Predicción de retraso con ML
   * @param {Object} features - Características de la entrega
   */
  async predictDelay(features) {
    const response = await apiClient.post('/predict', features)
    return response.data
  },

  /**
   * POST /chat - Consulta al agente Text-to-SQL
   * @param {string} pregunta - Pregunta en lenguaje natural
   */
  async chat(pregunta) {
    const response = await apiClient.post('/chat', { pregunta })
    return response.data
  },

  /**
   * GET /health - Health check de la API
   */
  async healthCheck() {
    const response = await apiClient.get('/health')
    return response.data
  },
}
