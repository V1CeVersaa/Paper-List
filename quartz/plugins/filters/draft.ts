import { QuartzFilterPlugin } from "../types"

export const RemoveDrafts: QuartzFilterPlugin<{}> = () => ({
  name: "RemoveDrafts",
  shouldPublish(_ctx, [_tree, vfile]) {
    const includePrivate =
      process.env.QUARTZ_PRIVATE === "1" || process.env.QUARTZ_PRIVATE === "true"
    const draftFlag =
      vfile.data?.frontmatter?.draft === true || vfile.data?.frontmatter?.draft === "true"
    const visibility =
      typeof vfile.data?.frontmatter?.visibility === "string"
        ? vfile.data.frontmatter.visibility.toLowerCase()
        : undefined

    if (visibility === "public") return true
    if (visibility === "local") return includePrivate
    if (visibility === "draft") return includePrivate

    if (draftFlag) return includePrivate
    return true
  },
})
