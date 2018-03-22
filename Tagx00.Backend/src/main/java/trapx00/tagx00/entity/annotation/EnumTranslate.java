package trapx00.tagx00.entity.annotation;

import java.lang.annotation.*;

@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.FIELD})
@Documented
public @interface EnumTranslate {
    Class targetClass();
}
