<template>
  <div class="bg-white rounded-lg border border-slate-200 overflow-hidden h-full flex flex-col">
    <div class="bg-blue-600 px-6 py-4 border-b border-blue-700">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 bg-blue-700 rounded-lg flex items-center justify-center">
          <BrainIcon class="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 class="text-lg font-bold text-white">Predictor de Retrasos</h2>
          <p class="text-sm text-blue-100">Machine Learning — RandomForest</p>
        </div>
      </div>
    </div>

    <div class="p-6 flex-1 overflow-y-auto">
      <form @submit.prevent="predecir" class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Distancia de la ruta (km)
          </label>
          <input
            v-model.number="form.distancia_km"
            type="number"
            step="0.1"
            min="1"
            max="5000"
            required
            class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
            placeholder="450.0"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Experiencia del conductor (años)
          </label>
          <input
            v-model.number="form.experiencia_anios"
            type="number"
            min="0"
            max="50"
            required
            class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
            placeholder="5"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Calificación del conductor (1-5)
          </label>
          <input
            v-model.number="form.calificacion"
            type="number"
            step="0.1"
            min="1"
            max="5"
            required
            class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
            placeholder="3.8"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Capacidad del vehículo (kg)
          </label>
          <input
            v-model.number="form.capacidad_kg"
            type="number"
            step="100"
            min="500"
            max="50000"
            required
            class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
            placeholder="8000"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Antigüedad del vehículo (años)
          </label>
          <input
            v-model.number="form.antiguedad_vehiculo"
            type="number"
            min="0"
            max="40"
            required
            class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
            placeholder="7"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Temperatura promedio (°C)
          </label>
          <input
            v-model.number="form.temp_promedio"
            type="number"
            step="0.5"
            min="-30"
            max="50"
            required
            class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
            placeholder="12.5"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Condición climática
          </label>
          <select
            v-model="form.condicion_clima"
            required
            class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
          >
            <option value="">Selecciona...</option>
            <option value="soleado">☀️ Soleado</option>
            <option value="lluvioso">🌧️ Lluvioso</option>
            <option value="nevado">❄️ Nevado</option>
            <option value="tormentoso">⛈️ Tormentoso</option>
          </select>
        </div>

        <button
          type="submit"
          :disabled="loading"
          class="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg border border-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-slate-400 flex items-center justify-center gap-2"
        >
          <SparklesIcon v-if="!loading" class="w-5 h-5" />
          <LoaderIcon v-else class="w-5 h-5 animate-spin" />
          {{ loading ? 'Prediciendo...' : 'Predecir Retraso' }}
        </button>
      </form>

      <div v-if="resultado" class="mt-6 p-4 rounded-lg border-2" :class="resultadoClasses">
        <div class="flex items-start gap-3">
          <component :is="resultadoIcon" class="w-6 h-6 shrink-0 mt-0.5" />
          <div class="flex-1">
            <p class="font-bold text-lg">{{ resultado.minutos_retraso_predicho.toFixed(1) }} minutos</p>
            <p class="text-sm opacity-90 mt-1">{{ resultado.mensaje }}</p>
            <div class="mt-3 inline-flex px-3 py-1.5 rounded-md text-xs font-bold uppercase tracking-wide" :class="nivelClasses">
              {{ resultado.nivel_riesgo }}
            </div>
          </div>
        </div>
      </div>

      <div v-if="error" class="mt-6 p-4 bg-red-50 border-2 border-red-300 rounded-lg">
        <div class="flex items-start gap-3">
          <AlertCircleIcon class="w-6 h-6 text-red-600 shrink-0" />
          <div>
            <p class="font-semibold text-red-900">Error al predecir</p>
            <p class="text-sm text-red-700 mt-1">{{ error }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { BrainIcon, SparklesIcon, LoaderIcon, AlertCircleIcon, CheckCircleIcon, AlertTriangleIcon, XCircleIcon } from 'lucide-vue-next'
import api from '../services/api'

const form = ref({
  distancia_km: 450,
  experiencia_anios: 5,
  calificacion: 3.8,
  capacidad_kg: 8000,
  antiguedad_vehiculo: 7,
  temp_promedio: 12.5,
  condicion_clima: 'lluvioso',
})

const loading = ref(false)
const resultado = ref(null)
const error = ref(null)

const predecir = async () => {
  loading.value = true
  error.value = null
  resultado.value = null

  try {
    const data = await api.predictDelay(form.value)
    resultado.value = data
  } catch (err) {
    error.value = err.response?.data?.detail || 'Error de conexión con la API'
  } finally {
    loading.value = false
  }
}

const resultadoClasses = computed(() => {
  if (!resultado.value) return ''
  const nivel = resultado.value.nivel_riesgo
  return {
    'bg-green-50 border-green-300': nivel === 'BAJO',
    'bg-yellow-50 border-yellow-300': nivel === 'MEDIO',
    'bg-red-50 border-red-300': nivel === 'ALTO',
  }
})

const nivelClasses = computed(() => {
  if (!resultado.value) return ''
  const nivel = resultado.value.nivel_riesgo
  return {
    'bg-green-100 text-green-700': nivel === 'BAJO',
    'bg-yellow-100 text-yellow-700': nivel === 'MEDIO',
    'bg-red-100 text-red-700': nivel === 'ALTO',
  }
})

const resultadoIcon = computed(() => {
  if (!resultado.value) return null
  const nivel = resultado.value.nivel_riesgo
  return nivel === 'BAJO' ? CheckCircleIcon : nivel === 'MEDIO' ? AlertTriangleIcon : XCircleIcon
})
</script>
