
export const isNativeASRAvailable = (): boolean => {
    if (typeof window === 'undefined') return false;
    return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
};

export const isNativeTTSAvailable = (): boolean => {
    if (typeof window === 'undefined') return false;
    return 'speechSynthesis' in window;
};

export const isWebGPUAvailable = async (): Promise<boolean> => {
    if (typeof navigator === 'undefined') return false;

    // Mobile check: WebGPU is often unstable on mobile browsers even if exposed
    const ua = navigator.userAgent.toLowerCase();
    const isMobile = /iphone|ipad|ipod|android|mobile/.test(ua);
    if (isMobile) {
        console.log("WebGPU disabled on mobile devices for stability");
        return false;
    }

    // @ts-ignore - navigator.gpu is not yet in all TS definitions
    if (!navigator.gpu) return false;

    try {
        // Check for adapter
        // @ts-ignore
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        // Check for device
        const device = await adapter.requestDevice();
        if (device) {
            device.destroy(); // Clean up
            return true;
        }
        return false;
    } catch (e) {
        console.warn("WebGPU detection failed:", e);
        return false;
    }
};
