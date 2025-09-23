#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>

#include <m_pd.h>
#include <g_canvas.h>

#include <onnx.h>

#define CURRENT_ONNX_OPSET 24
static t_class *onnx_tilde_class;

typedef struct _onnx_tilde {
    t_object obj;
    t_sample s;
    t_canvas *c;

    bool time_inference;

    bool supported;

    struct onnx_context_t *ctx;
    int tensor_out_count;
    t_symbol **tensor_outputs;
    t_inlet **ins;
    t_outlet **outs;
} t_onnx_tilde;

// ╭─────────────────────────────────────╮
// │           Object Helpers            │
// ╰─────────────────────────────────────╯
int onnx_tilde_check_compatibility(t_onnx_tilde *x) {
    if (!x || !x->ctx || !x->ctx->g) {
        pd_error(x, "[onnx~] Invalid context or graph");
        return 0;
    }
    struct onnx_graph_t *g = x->ctx->g;
    int opset = CURRENT_ONNX_OPSET;
    int all_ok = 1;
    for (int i = 0; i < g->nlen; i++) {
        struct onnx_node_t *n = &g->nodes[i];
        if (!n || !n->proto) {
            pd_error(x, "[onnx~] [UNSUPPORTED] (null-op) via (unresolved)");
            all_ok = 0;
            continue;
        }

        const char *domain =
            (n->proto->domain && n->proto->domain[0]) ? n->proto->domain : "ai.onnx";
        const char *op_type = n->proto->op_type ? n->proto->op_type : "(unknown)";
        const char *resolver = (n->r && n->r->name) ? n->r->name : "(unresolved)";
        int is_supported = (n->r != NULL);
        int version_ok = 1;
        if (opset > 0 && strcmp(domain, "ai.onnx") == 0) {
            version_ok = (n->opset <= opset);
        }
        const char *suffix = (!version_ok) ? " [NEWER-OPSET]" : "";
        const char *status = is_supported ? "[OK]" : "[UNSUPPORTED]";
        if (!is_supported || !version_ok) {
            pd_error(x, "[onnx~] %s %s-%d (%s) via %s%s", status, op_type, n->opset, domain,
                     resolver, suffix);
            all_ok = 0;
        }
    }
    return all_ok;
}

// ╭─────────────────────────────────────╮
// │           Object Methods            │
// ╰─────────────────────────────────────╯
static void onnx_tilde_set(t_onnx_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    if (argv[0].a_type != A_SYMBOL) {
        pd_error(x, "[onnx~] First argument must be a symbol, e.g., 'tensors'");
        return;
    }

    const char *method = atom_getsymbol(argv)->s_name;
    if (strcmp(method, "time_inference") == 0) {
        x->time_inference = argv[1].a_type == A_FLOAT && atom_getfloat(&argv[1]) > 0.0;
    }
}

// ─────────────────────────────────────
static void onnx_tilde_dump(t_onnx_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    if (!x || !x->ctx || !x->ctx->g) {
        pd_error(x, "[onnx~] Invalid context or graph");
        return;
    }

    if (argc < 1) {
        pd_error(x, "[onnx~] Invalid format");
        return;
    }

    if (argv[0].a_type != A_SYMBOL) {
        pd_error(x, "[onnx~] First argument must be a symbol, e.g., 'tensors'");
        return;
    }

    const char *method = atom_getsymbol(argv)->s_name;
    if (strcmp(method, "tensors_inputs") == 0) {
        struct onnx_graph_t *g = x->ctx->g;
        for (int i = 0; i < g->nlen; i++) {
            struct onnx_node_t *n = &g->nodes[i];
            if (!n || n->noutput <= 0) {
                continue;
            }
            for (int j = 0; j < n->ninput; j++) {
                onnx_tensor_t *t = n->inputs[j];
                if (!t) {
                    continue;
                }
                post("[onnx~] Tensor '%s'\n\tType: %s\n\tShape: %d\n", t->name ? t->name : "(null)",
                     onnx_tensor_type_tostring(t->type), t->ndim);
            }
        }
    } else if (strcmp(method, "tensors_outputs") == 0) {
        struct onnx_graph_t *g = x->ctx->g;
        for (int i = 0; i < g->nlen; i++) {
            struct onnx_node_t *n = &g->nodes[i];
            if (!n || n->noutput <= 0) {
                continue;
            }
            for (int j = 0; j < n->noutput; j++) {
                onnx_tensor_t *t = n->outputs[j];
                if (!t) {
                    continue;
                }
                post("[onnx~] Tensor '%s'\n\tType: %s\n\tShape: %d\n", t->name ? t->name : "(null)",
                     onnx_tensor_type_tostring(t->type), t->ndim);
            }
        }
    } else if (strcmp(method, "compatibility") == 0) {
        struct onnx_context_t *ctx = x->ctx;
        if (!ctx || !ctx->g || !ctx->model) {
            pd_error(x, "[onnx~] Invalid context / graph / model for compatibility check");
            return;
        }

        int opset = CURRENT_ONNX_OPSET;
        if (opset <= 0) {
            for (int i = 0; i < ctx->model->n_opset_import; i++) {
                const char *d = ctx->model->opset_import[i]->domain;
                if (!d || !d[0]) {
                    d = "ai.onnx";
                }
                if (strcmp(d, "ai.onnx") == 0) {
                    /* Protobuf version is int64_t; clamp to int if needed. */
                    long long v = (long long)ctx->model->opset_import[i]->version;
                    if (v > 0 && v < INT_MAX) {
                        opset = (int)v;
                    }
                    break;
                }
            }
        }
        post("[onnx~] Model IR version: %lld", (long long)ctx->model->ir_version);
        for (int i = 0; i < ctx->model->n_opset_import; i++) {
            const char *d = ctx->model->opset_import[i]->domain;
            if (!d || !d[0]) {
                d = "ai.onnx";
            }
            post("[onnx~] Opset import: %s = %lld", d,
                 (long long)ctx->model->opset_import[i]->version);
        }
        if (opset > 0) {
            post("[onnx~] Checking against CURRENT_ONNX_OPSET = %d", opset);
        } else {
            post("[onnx~] CURRENT_ONNX_OPSET not set; only reporting resolver support");
        }

        int total = 0, supported = 0, unsupported = 0, newer = 0;

        post("[onnx~] Checking Operators Compatibility", opset);
        struct onnx_graph_t *g = ctx->g;
        for (int i = 0; i < g->nlen; i++) {
            struct onnx_node_t *n = &g->nodes[i];
            if (!n || !n->proto) {
                continue;
            }
            total++;

            const char *domain =
                (n->proto->domain && n->proto->domain[0]) ? n->proto->domain : "ai.onnx";
            const char *op_type = n->proto->op_type ? n->proto->op_type : "(unknown)";
            const char *resolver = (n->r && n->r->name) ? n->r->name : "(unresolved)";

            int is_supported = (n->r != NULL);
            int version_ok = (opset <= 0) ? 1 : (n->opset <= opset);

            if (is_supported) {
                supported++;
            } else {
                unsupported++;
            }
            if (!version_ok) {
                newer++;
            }

            if (is_supported) {
                post("\t%s %s-%d (%s) via %s%s", is_supported ? "[OK]" : "[UNSUPPORTED]", op_type,
                     n->opset, domain, resolver, (!version_ok ? " [NEWER-OPSET]" : ""));
            } else {
                logpost(x, 1, "\t%s %s-%d (%s) via %s%s", is_supported ? "[OK]" : "[UNSUPPORTED]",
                        op_type, n->opset, domain, resolver, (!version_ok ? " [NEWER-OPSET]" : ""));
            }
        }

        post("[onnx~] Summary: total=%d, supported=%d, unsupported=%d, newer-opset=%d", total,
             supported, unsupported, newer);
    }
}

// ─────────────────────────────────────
static void onnx_tilde_anything(t_onnx_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    if (!x->supported) {
        pd_error(x, "[onnx~] Model loaded is not supported");
        return;
    }

    // Need at least the tensor name + 1 value
    if (argc < 2) {
        pd_error(x, "[onnx~] Expected a list starting with the tensor name followed by values, "
                    "e.g., 'mfcc 0.1 0.2 ...'");
        return;
    }
    if (argv[0].a_type != A_SYMBOL) {
        pd_error(x, "[onnx~] First item of list must be the tensor name (symbol)");
        return;
    }

    // Lookup input tensor by name
    const char *tname = atom_getsymbol(argv)->s_name;
    struct onnx_tensor_t *t = onnx_tensor_search(x->ctx, tname);
    if (!t) {
        pd_error(x, "[onnx~] Tensor '%s' not found in the model", tname);
        return;
    }

    // Ensure list length matches tensor element count
    const int nvals = argc - 1; // number of numeric items provided
    if ((size_t)nvals != t->ndata) {
        pd_error(x, "[onnx~] Tensor '%s' needs a list of exactly %zu numbers, got %d", t->name,
                 t->ndata, nvals);
        return;
    }

    // Only support float tensors for MFCC input
    switch (t->type) {
    case ONNX_TENSOR_TYPE_FLOAT32: {
        float *buf = (float *)malloc(sizeof(float) * (size_t)nvals);
        if (!buf) {
            pd_error(x, "[onnx~] Out of memory");
            return;
        }
        for (int i = 0; i < nvals; i++) {
            buf[i] = (float)atom_getfloat(argv + i + 1);
        }
        onnx_tensor_apply(t, (void *)buf, (size_t)nvals * sizeof(float));
        free(buf);
        break;
    }
    case ONNX_TENSOR_TYPE_FLOAT64: {
        double *buf = (double *)malloc(sizeof(double) * (size_t)nvals);
        if (!buf) {
            pd_error(x, "[onnx~] Out of memory");
            return;
        }
        for (int i = 0; i < nvals; i++) {
            buf[i] = (double)atom_getfloat(argv + i + 1);
        }
        onnx_tensor_apply(t, (void *)buf, (size_t)nvals * sizeof(double));
        free(buf);
        break;
    }
    default:
        pd_error(x,
                 "[onnx~] Unsupported input tensor type for '%s': %s (expected float32 or float64)",
                 t->name, onnx_tensor_type_tostring(t->type));
        return;
    }

    // Run the graph
    if (x->time_inference) {
        auto start = std::chrono::high_resolution_clock::now();
        onnx_run(x->ctx);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        post("Took %lld µs to run", static_cast<long long>(elapsed.count()));
    } else {
        onnx_run(x->ctx);
    }

    // Emit outputs
    for (int i = 0; i < x->tensor_out_count; i++) {
        struct onnx_tensor_t *to = onnx_tensor_search(x->ctx, x->tensor_outputs[i]->s_name);
        if (!to) {
            pd_error(x, "[onnx~] Output tensor '%s' not found", x->tensor_outputs[i]->s_name);
            continue;
        }
        if (to->ndata == 0) {
            pd_error(x, "[onnx~] Output tensor '%s' is empty", to->name);
            continue;
        }

        // allocate atoms dynamically to avoid huge stack frames
        t_atom *a = (t_atom *)malloc(sizeof(t_atom) * to->ndata);
        if (!a) {
            pd_error(x, "[onnx~] Out of memory preparing output '%s'", to->name);
            continue;
        }

        switch (to->type) {
        case ONNX_TENSOR_TYPE_FLOAT32: {
            float *data = (float *)to->datas;
            for (size_t j = 0; j < to->ndata; j++) {
                SETFLOAT(&a[j], (t_float)data[j]);
            }
            outlet_list(x->outs[i], gensym("list"), (int)to->ndata, a);
            break;
        }
        case ONNX_TENSOR_TYPE_FLOAT64: {
            double *data = (double *)to->datas;
            for (size_t j = 0; j < to->ndata; j++) {
                SETFLOAT(&a[j], (t_float)data[j]);
            }
            outlet_list(x->outs[i], gensym("list"), (int)to->ndata, a);
            break;
        }
        case ONNX_TENSOR_TYPE_INT32: {
            int32_t *data = (int32_t *)to->datas;
            for (size_t j = 0; j < to->ndata; j++) {
                SETFLOAT(&a[j], (t_float)data[j]);
            }
            outlet_list(x->outs[i], gensym("list"), (int)to->ndata, a);
            break;
        }
        case ONNX_TENSOR_TYPE_INT64: {
            int64_t *data = (int64_t *)to->datas;
            for (size_t j = 0; j < to->ndata; j++) {
                // Potential range loss, but Pd floats are fine for typical class counts/scores
                SETFLOAT(&a[j], (t_float)data[j]);
            }
            outlet_list(x->outs[i], gensym("list"), (int)to->ndata, a);
            break;
        }
        case ONNX_TENSOR_TYPE_STRING: {
            char **data = (char **)to->datas; // ONNX stores strings as char*[]
            t_atom *atoms = (t_atom *)malloc(sizeof(t_atom) * to->ndata);
            if (!atoms) {
                break;
            }

            for (size_t j = 0; j < to->ndata; j++) {
                if (data[j]) {
                    SETSYMBOL(&atoms[j], gensym(data[j]));
                } else {
                    SETSYMBOL(&atoms[j], gensym("")); // fallback for NULL strings
                }
            }

            outlet_list(x->outs[i], gensym("list"), (int)to->ndata, atoms);
            free(atoms);
            break;
        }

        default:
            pd_error(x, "[onnx~] Unsupported output tensor type for '%s': %s", to->name,
                     onnx_tensor_type_tostring(to->type));
            break;
        }

        free(a);
    }
}
// ─────────────────────────────────────
static void onnx_tilde_create_tensor_inout(t_onnx_tilde *x) {
    if (!x->ctx || !x->ctx->g) {
        return;
    }

    logpost(x, 3, "[onnx~] There is %d graphs", x->ctx->g->nlen);
    // create outputs tensors
}

// ─────────────────────────────────────
static void *onnx_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_onnx_tilde *x = (t_onnx_tilde *)pd_new(onnx_tilde_class);
    if (argc < 1) {
        pd_error(x, "[onnx~] needs at least the ONNX model filename");
        return NULL;
    }

    x->outs = NULL;

    t_symbol *model = atom_getsymbol(argv);
    char dirbuf[MAXPDSTRING], *nameptr;
    int fd = canvas_open(canvas_getcurrent(), model->s_name, "", dirbuf, &nameptr, MAXPDSTRING, 1);
    if (fd < 0) {
        pd_error(x, "[onnx~] could not find file %s", model->s_name);
        return NULL;
    }
    sys_close(fd);
    char fullpath[MAXPDSTRING];
    snprintf(fullpath, MAXPDSTRING, "%s/%s", dirbuf, nameptr);

    x->ctx = onnx_context_alloc_from_file(fullpath, NULL, 0);
    if (!x->ctx) {
        pd_error(x, "[onnx~] failed to load ONNX model: %s", fullpath);
        return NULL;
    } else {
        logpost(x, 3, "[onnx~] IR Version: v%ld", x->ctx->model->ir_version);
        logpost(x, 3, "[onnx~] Producer: %s %s", x->ctx->model->producer_name,
                x->ctx->model->producer_version);
        logpost(x, 3, "[onnx~] Domain: %s", x->ctx->model->domain);
        logpost(x, 3, "[onnx~] Imports: ");
        for (int i = 0; i < x->ctx->model->n_opset_import; i++) {
            logpost(x, 3, "\t\t%s v%ld",
                    (onnx_strlen(x->ctx->model->opset_import[i]->domain) > 0)
                        ? x->ctx->model->opset_import[i]->domain
                        : "ai.onnx",
                    x->ctx->model->opset_import[i]->version);
        }
    }
    // create input for tensors
    onnx_tilde_create_tensor_inout(x);
    if (x->ctx && x->ctx->g) {
        struct onnx_graph_t *g = x->ctx->g;
        for (int i = 0; i < g->nlen; i++) {
            struct onnx_node_t *n = &g->nodes[i];
            const char *domain = n->proto->domain;
            if (n->opset > CURRENT_ONNX_OPSET) {
                logpost(x, 1, "[onnx~] Unsupported opset => %s-%d (%s)\r\n", n->proto->op_type,
                        n->opset,
                        (onnx_strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx");
            }
        }
    }

    x->tensor_outputs = (t_symbol **)malloc((argc - 1) * sizeof(t_symbol *));
    x->outs = (t_outlet **)malloc((argc - 1) * sizeof(t_outlet *));
    if (!x->tensor_outputs || !x->outs) {
        pd_error(x, "[onnx~] Failed to allocate memory");
        return NULL;
    }

    for (int i = 1; i < argc; i++) {
        x->tensor_outputs[i - 1] = atom_getsymbol(argv + i);
        x->outs[i - 1] = outlet_new(&x->obj, &s_list);
    }
    x->tensor_out_count = argc - 1;

    x->supported = onnx_tilde_check_compatibility(x);
    x->time_inference = false;

    return x;
}

// ─────────────────────────────────────
static void onnx_tilde_free(t_onnx_tilde *x) {
    if (x->ctx) {
        onnx_context_free(x->ctx);
        x->ctx = NULL;
    }

    if (x->outs != NULL) {
        free(x->outs);
    }
}

// ─────────────────────────────────────
extern "C" void onnx_setup(void) {
    onnx_tilde_class = class_new(gensym("onnx"), (t_newmethod)onnx_tilde_new,
                                 (t_method)onnx_tilde_free, sizeof(t_onnx_tilde), 0, A_GIMME, 0);

    class_addanything(onnx_tilde_class, onnx_tilde_anything);
    class_addmethod(onnx_tilde_class, (t_method)onnx_tilde_set, gensym("set"), A_GIMME, A_NULL);
    class_addmethod(onnx_tilde_class, (t_method)onnx_tilde_dump, gensym("dump"), A_GIMME, A_NULL);
}
