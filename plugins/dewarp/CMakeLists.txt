add_library(gstdewarpfilter SHARED gstdewarpfilter.cpp)

# 链接 GStreamer 依赖
target_include_directories(gstdewarpfilter PRIVATE ${GST_INCLUDE_DIRS})
target_link_libraries(gstdewarpfilter PRIVATE ${GST_LIBRARIES})

# GStreamer 插件规范：去除 "lib" 前缀，生成 gstdewarpfilter.so 而不是 libgstdewarpfilter.so
set_target_properties(gstdewarpfilter PROPERTIES PREFIX "")
