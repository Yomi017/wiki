const { getGraph } = require("../../helpers/linkUtils");
const { getFileTree } = require("../../helpers/filetreeUtils");
const { userComputed } = require("../../helpers/userUtils");

module.exports = {
  graph: (data) => getGraph(data),
  filetree: (data) => getFileTree(data),
  // 添加动态文件树数据
  dynamicFileTree: (data) => {
    // 如果需要的话，这里可以添加额外的文件树处理逻辑
    return data.fileTree || [];
  },
  userComputed: (data) => userComputed(data),
};
