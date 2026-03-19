export class TtsStreamHandler {
  private buffer: string = '';

  processChunk(text: string, messageId: string, speakFn: (text: string, isFinal: boolean, id?: string) => void) {
    this.buffer += text;
    // Check for sentence delimiters
    // Note: 's' flag (dotAll) might not be supported in older targets, so using [\s\S] instead of . if needed
    // But actually, we just want to match up to the first delimiter.
    const delimiterRegex = /([.!?。！？\n])/;
    const parts = this.buffer.split(delimiterRegex);
    
    // If we have at least 3 parts (text + delimiter + rest), we have a complete sentence
    if (parts.length >= 2) {
        // parts[0] is text, parts[1] is delimiter. 
        // We might have multiple sentences.
        // Reconstruct sentences.
        let processedLength = 0;
        for (let i = 0; i < parts.length - 1; i += 2) {
            // Add a small pause punctuation if it was a sentence break
            // Comma might induce a shorter pause than Period in some TTS
            const sentence = parts[i] + (parts[i+1] || '');
            // Only speak if significant length
            if (sentence.trim().length > 1) {
                speakFn(sentence, false, messageId);
            }
            processedLength += sentence.length;
        }
        // Keep the remainder
        this.buffer = this.buffer.substring(processedLength);
    }
  }

  flush(messageId: string, speakFn: (text: string, isFinal: boolean, id?: string) => void) {
    if (this.buffer.trim().length > 0) {
        speakFn(this.buffer, false, messageId);
        this.buffer = '';
    }
  }

  reset() {
      this.buffer = '';
  }
}
