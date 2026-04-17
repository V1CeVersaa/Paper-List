import { FileTrieNode } from "./fileTrie"

export function createExplorerSortFn() {
  return function <T extends { slug: string; title: string; filePath: string }>(
    a: FileTrieNode<T>,
    b: FileTrieNode<T>,
  ) {
    if (a.isFolder !== b.isFolder) {
      return a.isFolder ? -1 : 1
    }

    if (!a.isFolder && !b.isFolder) {
      const aSpecialRank =
        a.data?.title === "Navigation"
          ? 0
          : a.data?.title === "Overview"
            ? 1
            : Number.POSITIVE_INFINITY
      const bSpecialRank =
        b.data?.title === "Navigation"
          ? 0
          : b.data?.title === "Overview"
            ? 1
            : Number.POSITIVE_INFINITY

      if (aSpecialRank !== bSpecialRank) {
        return aSpecialRank - bSpecialRank
      }
    }

    const aSegments = a.slug.split("/").filter(Boolean)
    const bSegments = b.slug.split("/").filter(Boolean)
    const aNormalizedSegments = a.isFolder ? aSegments.slice(0, -1) : aSegments
    const bNormalizedSegments = b.isFolder ? bSegments.slice(0, -1) : bSegments
    const aParent = aNormalizedSegments.slice(0, -1).join("/")
    const bParent = bNormalizedSegments.slice(0, -1).join("/")

    if (aParent === bParent) {
      const orderMap: Record<string, string[]> = {
        topics: [
          "reinforcement_learning",
          "imitation_learning",
          "active_imitation_learning",
          "preference_learning",
          "agentic_rl",
          "pomdp",
          "bandit_algorithms",
          "multimodal_reasoning",
          "textual_reasoning",
          "computer_vision",
        ],
      }
      const order = orderMap[aParent]

      if (order) {
        const aIndex = order.indexOf(aNormalizedSegments.at(-1) ?? "")
        const bIndex = order.indexOf(bNormalizedSegments.at(-1) ?? "")

        if (aIndex !== bIndex) {
          if (aIndex === -1) return 1
          if (bIndex === -1) return -1
          return aIndex - bIndex
        }
      }
    }

    return a.displayName.localeCompare(b.displayName, undefined, {
      numeric: true,
      sensitivity: "base",
    })
  }
}
