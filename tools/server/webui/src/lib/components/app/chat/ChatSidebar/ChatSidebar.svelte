<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { Trash2 } from '@lucide/svelte';
	import { ChatSidebarConversationItem, DialogConfirmation } from '$lib/components/app';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import Input from '$lib/components/ui/input/input.svelte';
	import {
		conversations,
		deleteConversation,
		updateConversationName
	} from '$lib/stores/chat.svelte';
	import { DatabaseStore } from '$lib/stores/database';
	import ChatSidebarActions from './ChatSidebarActions.svelte';

	const sidebar = Sidebar.useSidebar();

	let currentChatId = $derived(page.params.id);
	let isSearchModeActive = $state(false);
	let searchQuery = $state('');
	let showDeleteDialog = $state(false);
	let showEditDialog = $state(false);
	let showClearAllDialog = $state(false);
	let showMemoryDialog = $state(false);
	let selectedConversation = $state<DatabaseConversation | null>(null);
	let editedName = $state('');
	let memoryContent = $state('');
	let memoryLoading = $state(false);
	let memorySaving = $state(false);

	let filteredConversations = $derived.by(() => {
		if (searchQuery.trim().length > 0) {
			return conversations().filter((conversation: { name: string }) =>
				conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
			);
		}

		return conversations();
	});

	async function handleDeleteConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (conversation) {
			selectedConversation = conversation;
			showDeleteDialog = true;
		}
	}

	async function handleEditConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (conversation) {
			selectedConversation = conversation;
			editedName = conversation.name;
			showEditDialog = true;
		}
	}

	function handleConfirmDelete() {
		if (selectedConversation) {
			showDeleteDialog = false;

			setTimeout(() => {
				deleteConversation(selectedConversation.id);
				selectedConversation = null;
			}, 100); // Wait for animation to finish
		}
	}

	function handleConfirmEdit() {
		if (!editedName.trim() || !selectedConversation) return;

		showEditDialog = false;

		updateConversationName(selectedConversation.id, editedName);
		selectedConversation = null;
	}

	export function handleMobileSidebarItemClick() {
		if (sidebar.isMobile) {
			sidebar.toggle();
		}
	}

	export function activateSearchMode() {
		isSearchModeActive = true;
	}

	export function editActiveConversation() {
		if (currentChatId) {
			const activeConversation = filteredConversations.find((conv) => conv.id === currentChatId);

			if (activeConversation) {
				const event = new CustomEvent('edit-active-conversation', {
					detail: { conversationId: currentChatId }
				});
				document.dispatchEvent(event);
			}
		}
	}

	function handleClearAll() {
		showClearAllDialog = true;
	}

	async function handleConfirmClearAll() {
		showClearAllDialog = false;
		const count = await DatabaseStore.deleteAllConversations();
		console.log(`Cleared ${count} conversations`);
		
		// Navigate to home
		goto('/?new_chat=true#/');
		
		// Reload conversations
		window.location.reload();
	}

	async function loadMemory() {
		memoryLoading = true;
		try {
			const response = await fetch('/memory');
			const data = await response.json();
			memoryContent = data.content || '';
		} catch (error) {
			console.error('Failed to load memory:', error);
			memoryContent = '';
		} finally {
			memoryLoading = false;
		}
	}

	async function saveMemory() {
		memorySaving = true;
		try {
			const response = await fetch('/memory', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ content: memoryContent })
			});
			const data = await response.json();
			if (!data.success) {
				alert('Failed to save memory');
			}
		} catch (error) {
			console.error('Failed to save memory:', error);
			alert('Failed to save memory');
		} finally {
			memorySaving = false;
		}
	}

	async function handleOpenMemory() {
		showMemoryDialog = true;
		await loadMemory();
	}

	async function selectConversation(id: string) {
		if (isSearchModeActive) {
			isSearchModeActive = false;
			searchQuery = '';
		}

		await goto(`#/chat/${id}`);
	}
</script>

<ScrollArea class="h-[100vh]">
	<Sidebar.Header class=" top-0 z-10 gap-6 bg-sidebar/50 px-4 pt-4 pb-2 backdrop-blur-lg md:sticky">
		<a href="#/" onclick={handleMobileSidebarItemClick}>
			<h1 class="inline-flex items-center gap-1 px-2 text-xl font-semibold">llama.cpp</h1>
		</a>

		<ChatSidebarActions 
			{handleMobileSidebarItemClick} 
			bind:isSearchModeActive 
			bind:searchQuery
			onClearAll={handleClearAll}
			onOpenMemory={handleOpenMemory}
		/>
	</Sidebar.Header>

	<Sidebar.Group class="mt-4 space-y-2 p-0 px-4">
		{#if (filteredConversations.length > 0 && isSearchModeActive) || !isSearchModeActive}
			<Sidebar.GroupLabel>
				{isSearchModeActive ? 'Search results' : 'Conversations'}
			</Sidebar.GroupLabel>
		{/if}

		<Sidebar.GroupContent>
			<Sidebar.Menu>
				{#each filteredConversations as conversation (conversation.id)}
					<Sidebar.MenuItem class="mb-1">
						<ChatSidebarConversationItem
							conversation={{
								id: conversation.id,
								name: conversation.name,
								lastModified: conversation.lastModified,
								currNode: conversation.currNode
							}}
							{handleMobileSidebarItemClick}
							isActive={currentChatId === conversation.id}
							onSelect={selectConversation}
							onEdit={handleEditConversation}
							onDelete={handleDeleteConversation}
						/>
					</Sidebar.MenuItem>
				{/each}

				{#if filteredConversations.length === 0}
					<div class="px-2 py-4 text-center">
						<p class="mb-4 p-4 text-sm text-muted-foreground">
							{searchQuery.length > 0
								? 'No results found'
								: isSearchModeActive
									? 'Start typing to see results'
									: 'No conversations yet'}
						</p>
					</div>
				{/if}
			</Sidebar.Menu>
		</Sidebar.GroupContent>
	</Sidebar.Group>

	<div class="bottom-0 z-10 bg-sidebar bg-sidebar/50 px-4 py-4 backdrop-blur-lg md:sticky"></div>
</ScrollArea>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete Conversation"
	description={selectedConversation
		? `Are you sure you want to delete "${selectedConversation.name}"? This action cannot be undone and will permanently remove all messages in this conversation.`
		: ''}
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDelete}
	onCancel={() => {
		showDeleteDialog = false;
		selectedConversation = null;
	}}
/>

<AlertDialog.Root bind:open={showEditDialog}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Edit Conversation Name</AlertDialog.Title>
			<AlertDialog.Description>
				<Input
					class="mt-4 text-foreground"
					onkeydown={(e) => {
						if (e.key === 'Enter') {
							e.preventDefault();
							handleConfirmEdit();
						}
					}}
					placeholder="Enter a new name"
					type="text"
					bind:value={editedName}
				/>
			</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel
				onclick={() => {
					showEditDialog = false;
					selectedConversation = null;
				}}>Cancel</AlertDialog.Cancel
			>
			<AlertDialog.Action onclick={handleConfirmEdit}>Save</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>

<DialogConfirmation
	bind:open={showClearAllDialog}
	title="Clear All Conversations"
	description="Are you sure you want to delete ALL conversations? This action cannot be undone and will permanently remove all conversations and messages from your local database."
	confirmText="Clear All"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmClearAll}
	onCancel={() => {
		showClearAllDialog = false;
	}}
/>

<AlertDialog.Root bind:open={showMemoryDialog}>
	<AlertDialog.Content class="max-w-2xl max-h-[80vh]">
		<AlertDialog.Header>
			<AlertDialog.Title>💾 Memory Manager</AlertDialog.Title>
			<AlertDialog.Description>
				View and edit AI's persistent memory. Stored in <code>llama_memory.txt</code> (max 5KB).
			</AlertDialog.Description>
		</AlertDialog.Header>
		<div class="my-4">
			{#if memoryLoading}
				<div class="flex items-center justify-center p-8">
					<p class="text-sm text-muted-foreground">Loading memory...</p>
				</div>
			{:else}
				<div class="space-y-2">
					<p class="text-sm text-muted-foreground">
						The AI will read this memory at the start of each new conversation to personalize responses.
					</p>
					<textarea
						class="w-full rounded border p-3 bg-background min-h-[300px] max-h-[400px] font-mono text-sm"
						placeholder="Memory is empty. Add user preferences, name, language, etc.&#10;&#10;Example:&#10;User's name: John&#10;Language preference: English&#10;Context: Software engineer"
						bind:value={memoryContent}
					/>
					<div class="flex justify-between items-center text-xs text-muted-foreground">
						<span>{memoryContent.length} / 5120 bytes</span>
						{#if memoryContent.length > 5120}
							<span class="text-destructive">⚠️ Exceeds 5KB limit!</span>
						{/if}
					</div>
				</div>
			{/if}
		</div>
		<AlertDialog.Footer>
			<AlertDialog.Cancel
				onclick={() => {
					showMemoryDialog = false;
				}}>Cancel</AlertDialog.Cancel
			>
			<AlertDialog.Action
				onclick={async () => {
					await saveMemory();
					showMemoryDialog = false;
				}}
				disabled={memorySaving || memoryContent.length > 5120}
			>
				{memorySaving ? 'Saving...' : 'Save Memory'}
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
