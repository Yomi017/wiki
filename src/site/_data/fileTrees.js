const fs = require("fs");
const path = require("path");

module.exports = function () {
  const notesDir = path.join(__dirname, "../notes");

  function buildTree(dir, basePath = "") {
    const items = [];

    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const urlPath = path.join(basePath, entry.name).replace(/\\/g, "/");

        if (entry.isDirectory() && !entry.name.startsWith(".")) {
          items.push({
            type: "folder",
            name: entry.name,
            path: urlPath,
            children: buildTree(fullPath, urlPath),
          });
        } else if (entry.name.endsWith(".md")) {
          const nameWithoutExt = entry.name.replace(".md", "");
          // 读取文件的 front matter 来获取 permalink
          try {
            const fileContent = fs.readFileSync(fullPath, "utf8");
            const frontMatterMatch = fileContent.match(
              /^---\s*\n([\s\S]*?)\n---/
            );
            let permalink = `/${urlPath.replace(".md", "/").toLowerCase()}`;

            if (frontMatterMatch) {
              const frontMatter = frontMatterMatch[1];
              const permalinkMatch = frontMatter.match(/"permalink":"([^"]+)"/);
              if (permalinkMatch) {
                permalink = permalinkMatch[1];
              }
            }

            items.push({
              type: "file",
              name: nameWithoutExt,
              path: permalink,
            });
          } catch (error) {
            console.warn(`Could not read file: ${fullPath}`);
          }
        }
      }
    } catch (error) {
      console.warn(`Could not read directory: ${dir}`);
    }

    return items.sort((a, b) => {
      // 文件夹排在前面，然后按名称排序
      if (a.type === "folder" && b.type === "file") return -1;
      if (a.type === "file" && b.type === "folder") return 1;
      return a.name.localeCompare(b.name);
    });
  }

  return buildTree(notesDir);
};
