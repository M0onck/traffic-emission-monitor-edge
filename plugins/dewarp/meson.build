project('gst-hailo-dewarp', 'cpp', version : '1.0.0', default_options : ['cpp_std=c++17'])

# 1. 查找依赖
gst_dep = dependency('gstreamer-1.0')
gst_base_dep = dependency('gstreamer-base-1.0')
gst_video_dep = dependency('gstreamer-video-1.0')
gst_gl_dep = dependency('gstreamer-gl-1.0')

# 2. 收集源文件
plugin_sources = [
  'gstdewarpfilter.cpp'
]

# 3. 编译共享库 (.so) 名字必须严格匹配插件内部的宏
shared_library('gstdewarpfilter',
  plugin_sources,
  dependencies : [gst_dep, gst_base_dep, gst_video_dep, gst_gl_dep],
  install : true,
  install_dir : '/usr/lib/aarch64-linux-gnu/gstreamer-1.0'
)
