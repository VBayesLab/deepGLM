# Source all files within libs 02_folder
RootDir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(paste0(RootDir,'/02_libs'))

files.sources = list.files()
sapply(files.sources, source)

setwd(RootDir)


