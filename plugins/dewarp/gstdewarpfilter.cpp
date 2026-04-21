#include "gstdewarpfilter.h"
#include <stdio.h>
#include <stdlib.h>

#include <GLES3/gl3.h>
#include <gst/gl/gstglfuncs.h>

GST_DEBUG_CATEGORY_STATIC (gst_dewarp_filter_debug);
#define GST_CAT_DEFAULT gst_dewarp_filter_debug

/* ==============================================================
 * Pad 模板 (限制为 GPU 显存中的 RGBA 纹理)
 * ============================================================== */
#define CAPS_STR \
  "video/x-raw(memory:GLMemory), " \
  "format = (string) RGBA, " \
  "width = (int) [ 1, max ], " \
  "height = (int) [ 1, max ], " \
  "framerate = (fraction) [ 0, max ], " \
  "texture-target = (string) 2D"

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STR));

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STR));

/* ==============================================================
 * Shader 源码定义 
 * ============================================================== */

/* ==============================================================
 * 1. 顶点着色器 (使用 GLES 3.0 layout 强制锁定通道)
 * ============================================================== */
static const gchar *dewarp_vertex_source =
  "#version 300 es\n"
  "layout(location = 0) in vec4 a_position;\n"
  "layout(location = 1) in vec2 a_texcoord;\n"
  "out vec2 v_texcoord;\n"
  "void main() {\n"
  "   gl_Position = a_position;\n"
  "   v_texcoord = a_texcoord;\n"
  "}\n";

/* ==============================================================
 * 2. 片元着色器 (恢复纯净版)
 * ============================================================== */
static const gchar *dewarp_fragment_source =
  "#version 300 es\n"
  "precision highp float;\n"
  "in vec2 v_texcoord;\n"
  "uniform sampler2D tex;\n"
  "uniform sampler2D map_tex;\n"
  "out vec4 fragColor;\n"
  "void main() {\n"
  "  vec4 packed_uv = texture(map_tex, v_texcoord);\n"
  "  float src_x = packed_uv.r + (packed_uv.g / 255.0);\n"
  "  float src_y = packed_uv.b + (packed_uv.a / 255.0);\n"
  "\n"
  "  // 边缘越界裁切 (显示黑边)\n"
  "  if (src_x <= 0.001 || src_x >= 0.999 || src_y <= 0.001 || src_y >= 0.999) {\n"
  "      fragColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
  "  } else {\n"
  "      float vid_y = 1.0 - src_y;\n"
  "      fragColor = texture(tex, vec2(src_x, vid_y));\n"
  "  }\n"
  "}\n";

enum
{
  PROP_0,
  PROP_MAP_FILE_PATH,
  PROP_MAP_WIDTH,
  PROP_MAP_HEIGHT
};

G_DEFINE_TYPE_WITH_CODE (GstDewarpFilter, gst_dewarp_filter, GST_TYPE_GL_FILTER,
    GST_DEBUG_CATEGORY_INIT (gst_dewarp_filter_debug, "dewarpfilter", 0, "GPU Image Dewarp Filter"));

static void gst_dewarp_filter_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_dewarp_filter_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_dewarp_filter_finalize (GObject * object);
static gboolean gst_dewarp_filter_gl_start (GstGLBaseFilter * base_filter);
static void gst_dewarp_filter_gl_stop (GstGLBaseFilter * base_filter);
static gboolean gst_dewarp_filter_filter_texture (GstGLFilter * filter, GstGLMemory * in_tex, GstGLMemory * out_tex);

static void
gst_dewarp_filter_class_init (GstDewarpFilterClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstGLBaseFilterClass *glbase_filter_class = GST_GL_BASE_FILTER_CLASS (klass);
  GstGLFilterClass *gl_filter_class = GST_GL_FILTER_CLASS (klass);

  gobject_class->set_property = gst_dewarp_filter_set_property;
  gobject_class->get_property = gst_dewarp_filter_get_property;
  gobject_class->finalize = gst_dewarp_filter_finalize;

  g_object_class_install_property (gobject_class, PROP_MAP_FILE_PATH,
      g_param_spec_string ("map-file-path", "Map File Path",
          "Path to the binary RGBA mapping file", NULL,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MAP_WIDTH,
      g_param_spec_int ("map-width", "Map Width",
          "Width of the mapping table", 1, 8192, 1280,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MAP_HEIGHT,
      g_param_spec_int ("map-height", "Map Height",
          "Height of the mapping table", 1, 8192, 720,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_metadata (element_class,
      "GPU Dewarp Filter RGBA", "Filter/Effect/Video",
      "Dewarps video using a precomputed RGBA map on the GPU",
      "Developer");

  gst_element_class_add_static_pad_template (element_class, &src_factory);
  gst_element_class_add_static_pad_template (element_class, &sink_factory);

  glbase_filter_class->gl_start = gst_dewarp_filter_gl_start;
  glbase_filter_class->gl_stop = gst_dewarp_filter_gl_stop;
  gl_filter_class->filter_texture = gst_dewarp_filter_filter_texture;
}

static void gst_dewarp_filter_init (GstDewarpFilter * self) {
  self->map_file_path = NULL;
  self->map_width = 1280;
  self->map_height = 720;
  self->map_loaded = FALSE;
  self->shader = NULL;
  self->map_texture_id = 0;
}

static void gst_dewarp_filter_finalize (GObject * object) {
  GstDewarpFilter *self = GST_DEWARP_FILTER (object);
  g_free (self->map_file_path);
  G_OBJECT_CLASS (gst_dewarp_filter_parent_class)->finalize (object);
}

static void gst_dewarp_filter_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec) {
  GstDewarpFilter *self = GST_DEWARP_FILTER (object);
  switch (prop_id) {
    case PROP_MAP_FILE_PATH:
      g_free (self->map_file_path);
      self->map_file_path = g_value_dup_string (value);
      break;
    case PROP_MAP_WIDTH:
      self->map_width = g_value_get_int (value);
      break;
    case PROP_MAP_HEIGHT:
      self->map_height = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void gst_dewarp_filter_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec) {
  GstDewarpFilter *self = GST_DEWARP_FILTER (object);
  switch (prop_id) {
    case PROP_MAP_FILE_PATH:
      g_value_set_string (value, self->map_file_path);
      break;
    case PROP_MAP_WIDTH:
      g_value_set_int (value, self->map_width);
      break;
    case PROP_MAP_HEIGHT:
      g_value_set_int (value, self->map_height);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean gst_dewarp_filter_gl_start (GstGLBaseFilter * base_filter) {
  GstDewarpFilter *self = GST_DEWARP_FILTER (base_filter);
  GstGLContext *context = base_filter->context;
  const GstGLFuncs *gl = context->gl_vtable;
  GError *error = NULL;

  GstGLSLStage *vert_stage = gst_glsl_stage_new_with_string (context,
      GL_VERTEX_SHADER, GST_GLSL_VERSION_NONE, GST_GLSL_PROFILE_NONE, dewarp_vertex_source);
  
  GstGLSLStage *frag_stage = gst_glsl_stage_new_with_string (context,
      GL_FRAGMENT_SHADER, GST_GLSL_VERSION_NONE, GST_GLSL_PROFILE_NONE, dewarp_fragment_source);

  if (!vert_stage || !frag_stage) {
    GST_ERROR_OBJECT (self, "无法创建 Shader 阶段对象。");
    if (vert_stage) gst_object_unref (vert_stage);
    if (frag_stage) gst_object_unref (frag_stage);
    return FALSE;
  }

  self->shader = gst_gl_shader_new (context);

  if (!gst_gl_shader_compile_attach_stage (self->shader, vert_stage, &error)) {
    GST_ERROR_OBJECT (self, "顶点着色器编译失败: %s", error->message);
    g_clear_error (&error);
    gst_object_unref (vert_stage);
    gst_object_unref (frag_stage);
    gst_object_unref (self->shader);
    self->shader = NULL;
    return FALSE;
  }

  if (!gst_gl_shader_compile_attach_stage (self->shader, frag_stage, &error)) {
    GST_ERROR_OBJECT (self, "片元着色器编译失败: %s", error->message);
    g_clear_error (&error);
    gst_object_unref (vert_stage);
    gst_object_unref (frag_stage);
    gst_object_unref (self->shader);
    self->shader = NULL;
    return FALSE;
  }

  if (!gst_gl_shader_link (self->shader, &error)) {
    GST_ERROR_OBJECT (self, "Shader 链接失败: %s", error->message);
    g_clear_error (&error);
    gst_object_unref (vert_stage);
    gst_object_unref (frag_stage);
    gst_object_unref (self->shader);
    self->shader = NULL;
    return FALSE;
  }

  // 加载 RGBA 二进制映射表到 GPU 纹理
  if (self->map_file_path && !self->map_loaded) {
    FILE *fp = fopen (self->map_file_path, "rb");
    if (fp) {
      // RGBA8888 格式，每个像素 4 个字节
      size_t map_size = (size_t)self->map_width * self->map_height * 4; 
      uint8_t *map_data = (uint8_t *)malloc (map_size);
      
      if (fread (map_data, 1, map_size, fp) == map_size) {
        gl->GenTextures (1, &self->map_texture_id);
        gl->BindTexture (GL_TEXTURE_2D, self->map_texture_id);
        
        // 强制 1 字节对齐，防止跨平台内存行对齐问题
        gl->PixelStorei(GL_UNPACK_ALIGNMENT, 1);
        
        // 使用标准的 GL_RGBA8 加载纹理，彻底避开浮点限制
        gl->TexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, self->map_width, self->map_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, map_data);
        
        // 使用最近邻插值，防止 GPU 试图在 8-bit 被拆分的像素间插入错误值
        gl->TexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl->TexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        gl->TexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        gl->TexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        self->map_loaded = TRUE;
        GST_INFO_OBJECT (self, "成功加载 RGBA 映射表到 GPU 显存！");
      } else {
        GST_ERROR_OBJECT (self, "读取映射文件失败，文件尺寸不匹配！预期: %zu 字节", map_size);
      }
      free (map_data);
      fclose (fp);
    } else {
      GST_ERROR_OBJECT (self, "无法打开映射文件路径: %s", self->map_file_path);
    }
  }

  return TRUE;
}

static void gst_dewarp_filter_gl_stop (GstGLBaseFilter * base_filter) {
  GstDewarpFilter *self = GST_DEWARP_FILTER (base_filter);
  GstGLContext *context = base_filter->context;
  const GstGLFuncs *gl = context->gl_vtable;

  if (self->shader) {
    gst_object_unref (self->shader);
    self->shader = NULL;
  }

  if (self->map_loaded && self->map_texture_id) {
    gl->DeleteTextures (1, &self->map_texture_id);
    self->map_texture_id = 0;
    self->map_loaded = FALSE;
  }
}

/* =======================================================
 * 自定义渲染回调函数
 * ======================================================= */
static gboolean
_draw_dewarp_cb (GstGLFilter * filter, GstGLMemory * in_tex, gpointer user_data)
{
  GstDewarpFilter *self = GST_DEWARP_FILTER (user_data);
  GstGLContext *context = GST_GL_BASE_FILTER (filter)->context;
  const GstGLFuncs *gl = context->gl_vtable;

  gst_gl_shader_use (self->shader);

  // 手动向 GStreamer 注册顶点属性位置
  // 必须告诉 GStreamer 把全屏顶点送进我们 Shader 的 0 号和 1 号通道
  filter->draw_attr_position_loc = 0;
  filter->draw_attr_texture_loc = 1;

  // 绑定 RGBA 畸变映射表 (纹理 1)
  gl->ActiveTexture (GL_TEXTURE1);
  gl->BindTexture (GL_TEXTURE_2D, self->map_texture_id);
  gst_gl_shader_set_uniform_1i (self->shader, "map_tex", 1);

  // 绑定实时的视频输入画面 (纹理 0)
  gl->ActiveTexture (GL_TEXTURE0);
  gl->BindTexture (GL_TEXTURE_2D, gst_gl_memory_get_texture_id (in_tex));
  gst_gl_shader_set_uniform_1i (self->shader, "tex", 0);

  // 提交全屏绘制指令
  gst_gl_filter_draw_fullscreen_quad (filter);

  return TRUE;
}

/* =======================================================
 * 入口过滤函数
 * ======================================================= */
static gboolean 
gst_dewarp_filter_filter_texture (GstGLFilter * filter, GstGLMemory * in_tex, GstGLMemory * out_tex) 
{
  GstDewarpFilter *self = GST_DEWARP_FILTER (filter);

  if (!self->map_loaded || !self->shader) {
    // 如果还没加载完毕，直接放行原视频流
    gst_gl_filter_render_to_target_with_shader (filter, in_tex, out_tex, NULL);
    return TRUE;
  }

  // 将渲染和 FBO 管理全部交给 GStreamer 的回调安全作用域
  gst_gl_filter_render_to_target (filter, in_tex, out_tex, _draw_dewarp_cb, self);

  return TRUE;
}

static gboolean plugin_init (GstPlugin * plugin) {
  return gst_element_register (plugin, "dewarpfilter", GST_RANK_NONE, GST_TYPE_DEWARP_FILTER);
}

#ifndef VERSION
#define VERSION "1.0.0"
#endif
#ifndef PACKAGE
#define PACKAGE "gst-hailo-dewarp"
#endif

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    dewarpfilter,
    "GPU Accelerated Image Dewarping using RGBA map",
    plugin_init, VERSION, "LGPL", "Traffic Emission Monitor Edge", "https://github.com/m0onck/traffic-emission-monitor-edge"
)
