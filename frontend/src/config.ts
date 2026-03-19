export const config = {
  tts: {
    backend: 'auto' as 'native' | 'webgpu' | 'doubao' | 'auto',
    webgpu: {
      zh: {
        model: 'sherpa',
        mode: 'streaming',
      },
      en: {
        model: 'kokoro',
        mode: 'streaming',
      },
    },
    doubao: {
      voice: 'zh_female_vv_uranus_bigtts',
    },
  },
};
