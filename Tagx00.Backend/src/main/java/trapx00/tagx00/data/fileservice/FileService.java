package trapx00.tagx00.data.fileservice;

public interface FileService<T> {

    /**
     * save the entity
     *
     * @param entity the entity object
     * @return the entity if success else return null
     */
     T saveTuple(T entity);

    /**
     * find a entity
     *
     * @param info the key info to find
     * @return the entity
     */
    T findOne(String info, Class<T> clazz);

    void delete(String id, Class<T> clazz);


    T[] findOnes(String info, Class<T> clazz);
}

