export type ContentSegment =
	| { type: 'text'; content: string }
	| { type: 'thinking'; content: string; isClosed: boolean };

/**
 * Parse content into segments preserving order of text and thinking blocks
 * Returns array of segments for proper rendering
 */
export function parseInterThinkContent(content: string): ContentSegment[] {
	if (!content) {
		return [];
	}

	const segments: ContentSegment[] = [];
	const regex = /<inter_think>([\s\S]*?)(<\/inter_think>|$)/g;
	
	let lastIndex = 0;
	let match: RegExpExecArray | null;
	
	while ((match = regex.exec(content)) !== null) {
		// Add text before this thinking block
		if (match.index > lastIndex) {
			const textContent = content.slice(lastIndex, match.index).trim();
			if (textContent) {
				segments.push({ type: 'text', content: textContent });
			}
		}
		
		// Add thinking block
		const thinkingContent = match[1].trim();
		const isClosed = match[2] === '</inter_think>';
		if (thinkingContent) {
			segments.push({ 
				type: 'thinking', 
				content: thinkingContent,
				isClosed 
			});
		}
		
		lastIndex = match.index + match[0].length;
	}
	
	// Add remaining text after last thinking block
	if (lastIndex < content.length) {
		const textContent = content.slice(lastIndex).trim();
		if (textContent) {
			segments.push({ type: 'text', content: textContent });
		}
	}
	
	return segments;
}

/**
 * Legacy function for backward compatibility
 * Extract <inter_think> blocks from content
 * Returns {thinking: string | null, content: string}
 */
export function extractInterThinkBlocks(content: string): {
	thinking: string | null;
	content: string;
} {
	const segments = parseInterThinkContent(content);
	
	const thinkingParts = segments
		.filter((s): s is ContentSegment & { type: 'thinking' } => s.type === 'thinking')
		.map(s => s.content);
	
	const textParts = segments
		.filter((s): s is ContentSegment & { type: 'text' } => s.type === 'text')
		.map(s => s.content);
	
	return {
		thinking: thinkingParts.length > 0 ? thinkingParts.join('\n\n') : null,
		content: textParts.join('\n\n')
	};
}
