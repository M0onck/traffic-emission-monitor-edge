#ifndef __GST_DEWARP_FILTER_H__
#define __GST_DEWARP_FILTER_H__

#include <gst/gst.h>
#include <gst/gl/gl.h>
#include <gst/gl/gstglfilter.h>

G_BEGIN_DECLS

#define GST_TYPE_DEWARP_FILTER (gst_dewarp_filter_get_type())
G_DECLARE_FINAL_TYPE (GstDewarpFilter, gst_dewarp_filter, GST, DEWARP_FILTER, GstGLFilter)

struct _GstDewarpFilter
{
  GstGLFilter filter;

  gchar *map_file_path;
  gint map_width;
  gint map_height;

  gboolean map_loaded;
  GstGLShader *shader;
  guint map_texture_id; // 修改为 guint
};

G_END_DECLS

#endif /* __GST_DEWARP_FILTER_H__ */
