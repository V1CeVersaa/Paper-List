import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/V1CeVersaa/Paper-List",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
        { Component: Component.ReaderMode() },
      ],
    }),
    Component.Explorer({
      folderDefaultState: "open",
      defaultOpenDepth: 1,
      sortFn: (a, b) => {
        if ((!a.isFolder && !b.isFolder) || (a.isFolder && b.isFolder)) {
          if (!a.isFolder && !b.isFolder) {
            const aIsNav = a.data?.title === "Navigation"
            const bIsNav = b.data?.title === "Navigation"
            if (aIsNav && !bIsNav) return -1
            if (!aIsNav && bIsNav) return 1
          }
          return a.displayName.localeCompare(b.displayName, undefined, {
            numeric: true,
            sensitivity: "base",
          })
        }

        if (!a.isFolder && b.isFolder) {
          return 1
        } else {
          return -1
        }
      },
      mapFn: (node) => {
        if (node.displayName === "Navigation") {
          node.displayName = "🧭 Navigation"
        }
        return node
      },
    }),
  ],
  right: [
    // Component.Graph(),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer({
      folderDefaultState: "open",
      defaultOpenDepth: 1,
      sortFn: (a, b) => {
        if ((!a.isFolder && !b.isFolder) || (a.isFolder && b.isFolder)) {
          if (!a.isFolder && !b.isFolder) {
            const aIsNav = a.data?.title === "Navigation"
            const bIsNav = b.data?.title === "Navigation"
            if (aIsNav && !bIsNav) return -1
            if (!aIsNav && bIsNav) return 1
          }
          return a.displayName.localeCompare(b.displayName, undefined, {
            numeric: true,
            sensitivity: "base",
          })
        }

        if (!a.isFolder && b.isFolder) {
          return 1
        } else {
          return -1
        }
      },
      mapFn: (node) => {
        if (node.displayName === "Navigation") {
          node.displayName = "🧭 Navigation"
        }
        return node
      },
    }),
  ],
  right: [Component.DesktopOnly(Component.TableOfContents())],
}
