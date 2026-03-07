<template>
  <div class="bg-white rounded-lg border border-slate-200 overflow-hidden h-full flex flex-col">
    <div class="bg-slate-900 px-6 py-4 border-b border-slate-800">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center">
          <MessageSquareIcon class="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 class="text-lg font-bold text-white">Asistente IA</h2>
          <p class="text-sm text-slate-300">Text-to-SQL — Groq Llama 3.3</p>
        </div>
      </div>
    </div>

    <div ref="messagesContainer" class="flex-1 overflow-y-auto p-6 space-y-4 bg-slate-50">
      <div v-if="mensajes.length === 0" class="text-center py-12">
        <div class="inline-flex w-16 h-16 bg-slate-100 border-2 border-slate-200 rounded-lg items-center justify-center mb-4">
          <SparklesIcon class="w-8 h-8 text-slate-600" />
        </div>
        <h3 class="text-lg font-bold text-slate-900 mb-2">¡Hola! Soy tu asistente logístico</h3>
        <p class="text-sm text-slate-600 max-w-sm mx-auto font-medium">
          Puedo ayudarte a consultar el Data Warehouse. Pregúntame sobre conductores, rutas, clima o entregas.
        </p>
        <div class="mt-6 flex flex-wrap justify-center gap-2">
          <button
            v-for="ejemplo in ejemplos"
            :key="ejemplo"
            @click="preguntaInput = ejemplo"
            class="px-3 py-2 text-xs font-medium bg-white hover:bg-slate-100 border border-slate-300 rounded-lg transition"
          >
            {{ ejemplo }}
          </button>
        </div>
      </div>

      <div
        v-for="(msg, idx) in mensajes"
        :key="idx"
        class="flex gap-3"
        :class="msg.tipo === 'usuario' ? 'justify-end' : 'justify-start'"
      >
        <div
          v-if="msg.tipo === 'asistente'"
          class="w-8 h-8 rounded-lg shrink-0 flex items-center justify-center"
          :class="msg.error ? 'bg-red-100 border border-red-200' : 'bg-slate-700 border border-slate-600'"
        >
          <BotIcon v-if="!msg.error" class="w-5 h-5 text-white" />
          <AlertCircleIcon v-else class="w-5 h-5 text-red-600" />
        </div>

        <div
          class="max-w-[75%] rounded-lg px-4 py-3 border"
          :class="
            msg.tipo === 'usuario'
              ? 'bg-blue-600 border-blue-700 text-white'
              : msg.error
              ? 'bg-red-50 border-red-300 text-red-900'
              : 'bg-white border-slate-200 text-slate-900'
          "
        >
          <p class="text-sm whitespace-pre-wrap leading-relaxed">{{ msg.texto }}</p>
          <p v-if="msg.tipo === 'asistente' && !msg.error" class="text-xs opacity-60 mt-2">
            {{ new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' }) }}
          </p>
        </div>

        <div
          v-if="msg.tipo === 'usuario'"
          class="w-8 h-8 bg-blue-600 border border-blue-700 rounded-lg shrink-0 flex items-center justify-center text-white font-bold text-sm"
        >
          U
        </div>
      </div>

      <div v-if="loading" class="flex gap-3 justify-start">
        <div class="w-8 h-8 bg-slate-700 border border-slate-600 rounded-lg flex items-center justify-center">
          <LoaderIcon class="w-5 h-5 text-white animate-spin" />
        </div>
        <div class="bg-white border border-slate-200 rounded-lg px-4 py-3">
          <div class="flex gap-1">
            <span class="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></span>
            <span class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></span>
            <span class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></span>
          </div>
        </div>
      </div>
    </div>

    <div class="p-4 bg-white border-t border-slate-200">
      <form @submit.prevent="enviarPregunta" class="flex gap-2">
        <input
          v-model="preguntaInput"
          type="text"
          placeholder="Escribe tu pregunta sobre los datos..."
          :disabled="loading"
          class="flex-1 px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          :disabled="loading || !preguntaInput.trim()"
          class="px-6 py-2.5 bg-slate-900 hover:bg-slate-800 text-white font-semibold rounded-lg border border-slate-800 transition disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-slate-400 flex items-center gap-2"
        >
          <SendIcon class="w-5 h-5" />
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { MessageSquareIcon, BotIcon, SparklesIcon, SendIcon, LoaderIcon, AlertCircleIcon } from 'lucide-vue-next'
import api from '../services/api'

const preguntaInput = ref('')
const mensajes = ref([])
const loading = ref(false)
const messagesContainer = ref(null)

const ejemplos = [
  '¿Cuántas entregas hubo con clima tormentoso?',
  '¿Cuál es el conductor con mayor retraso promedio?',
  'Muestra las 5 rutas más largas',
]

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

const enviarPregunta = async () => {
  if (!preguntaInput.value.trim() || loading.value) return

  const pregunta = preguntaInput.value.trim()
  
  mensajes.value.push({
    tipo: 'usuario',
    texto: pregunta,
  })
  
  preguntaInput.value = ''
  loading.value = true
  scrollToBottom()

  try {
    const data = await api.chat(pregunta)
    
    mensajes.value.push({
      tipo: 'asistente',
      texto: data.respuesta,
      error: false,
    })
  } catch (err) {
    const errorMsg = err.response?.data?.detail || 'Error de conexión con la API. Verifica que el backend esté corriendo.'
    
    mensajes.value.push({
      tipo: 'asistente',
      texto: errorMsg,
      error: true,
    })
  } finally {
    loading.value = false
    scrollToBottom()
  }
}
</script>
