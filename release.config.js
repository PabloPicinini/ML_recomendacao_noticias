module.exports = {
  branches: ["main"],            // diz que a única branch de release é a main
  repositoryUrl: "https://github.com/PabloPicinini/ML_recomendacao_noticias.git", // link do seu repositório
  plugins: [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/git",
    "@semantic-release/github"
  ]
};