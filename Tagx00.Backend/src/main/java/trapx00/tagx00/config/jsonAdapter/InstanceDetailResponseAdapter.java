package trapx00.tagx00.config.jsonAdapter;

import com.google.gson.JsonElement;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import trapx00.tagx00.config.GsonFactory;
import trapx00.tagx00.vo.mission.instance.InstanceDetailVo;

import java.lang.reflect.Type;

public class InstanceDetailResponseAdapter implements JsonSerializer<InstanceDetailVo> {
    /**
     * Gson invokes this call-back method during serialization when it encounters a field of the
     * specified type.
     * <p>
     * <p>In the implementation of this call-back method, you should consider invoking
     * {@link JsonSerializationContext#serialize(Object, Type)} method to create JsonElements for any
     * non-trivial field of the {@code src} object. However, you should never invoke it on the
     * {@code src} object itself since that will cause an infinite loop (Gson will call your
     * call-back method again).</p>
     *
     * @param src       the object that needs to be converted to Json.
     * @param typeOfSrc the actual type (fully genericized version) of the source object.
     * @param context
     * @return a JsonElement corresponding to the specified object.
     */
    @Override
    public JsonElement serialize(InstanceDetailVo src, Type typeOfSrc, JsonSerializationContext context) {
        return new JsonPrimitive(GsonFactory.get().toJson(src, src.getMissionType().getInstanceDetailVoClass()));
    }
}
