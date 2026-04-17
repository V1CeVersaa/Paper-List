import assert from "node:assert"
import test, { describe } from "node:test"
import { createExplorerSortFn } from "./explorerSort"
import { FileTrieNode } from "./fileTrie"
import { FullSlug } from "./path"

interface TestData {
  title: string
  slug: string
  filePath: string
}

describe("explorerSortFn", () => {
  const explorerSortFn = createExplorerSortFn()

  test("survives the explorer string serialization round-trip", () => {
    const revivedSortFn = new Function(`return ${explorerSortFn.toString()}`)() as typeof explorerSortFn

    const trie = FileTrieNode.fromEntries<TestData>([
      [
        "topics/active_imitation_learning/index" as FullSlug,
        {
          title: "Z Active",
          slug: "topics/active_imitation_learning/index",
          filePath: "topics/active_imitation_learning/index.md",
        },
      ],
      [
        "topics/imitation_learning/index" as FullSlug,
        {
          title: "A Imitation",
          slug: "topics/imitation_learning/index",
          filePath: "topics/imitation_learning/index.md",
        },
      ],
    ])

    const topicsNode = trie.findNode(["topics"])
    assert.ok(topicsNode)

    topicsNode.sort(revivedSortFn)
    assert.deepStrictEqual(
      topicsNode.children.map((node) => node.slugSegment),
      ["imitation_learning", "active_imitation_learning"],
    )
  })

  test("keeps configured topic slug order even when titles would sort differently", () => {
    const trie = FileTrieNode.fromEntries<TestData>([
      [
        "topics/reinforcement_learning/index" as FullSlug,
        {
          title: "Z Topic",
          slug: "topics/reinforcement_learning/index",
          filePath: "topics/reinforcement_learning/index.md",
        },
      ],
      [
        "topics/imitation_learning/index" as FullSlug,
        {
          title: "A Topic",
          slug: "topics/imitation_learning/index",
          filePath: "topics/imitation_learning/index.md",
        },
      ],
    ])

    const topicsNode = trie.findNode(["topics"])
    assert.ok(topicsNode)

    topicsNode.sort(explorerSortFn)
    assert.deepStrictEqual(
      topicsNode.children.map((node) => node.slugSegment),
      ["reinforcement_learning", "imitation_learning"],
    )
  })

  test("keeps navigation and overview ahead of ordinary files", () => {
    const trie = FileTrieNode.fromEntries<TestData>([
      [
        "topics/reinforcement_learning/TRPO" as FullSlug,
        {
          title: "TRPO",
          slug: "topics/reinforcement_learning/TRPO",
          filePath: "topics/reinforcement_learning/TRPO.md",
        },
      ],
      [
        "topics/reinforcement_learning/overview" as FullSlug,
        {
          title: "Overview",
          slug: "topics/reinforcement_learning/overview",
          filePath: "topics/reinforcement_learning/overview.md",
        },
      ],
      [
        "topics/reinforcement_learning/navigation" as FullSlug,
        {
          title: "Navigation",
          slug: "topics/reinforcement_learning/navigation",
          filePath: "topics/reinforcement_learning/navigation.md",
        },
      ],
    ])

    const rlNode = trie.findNode(["topics", "reinforcement_learning"])
    assert.ok(rlNode)

    rlNode.sort(explorerSortFn)
    assert.deepStrictEqual(
      rlNode.children.map((node) => node.displayName),
      ["Navigation", "Overview", "TRPO"],
    )
  })
})
