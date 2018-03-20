package trapx00.tagx00.data.fileservice;

import net.sf.json.JSONArray;
import net.sf.json.JSONObject;
import net.sf.json.JsonConfig;
import org.springframework.stereotype.Service;
import trapx00.tagx00.entity.Entity;
import trapx00.tagx00.entity.annotation.Column;
import trapx00.tagx00.entity.annotation.ElementCollection;
import trapx00.tagx00.entity.annotation.Id;
import trapx00.tagx00.entity.annotation.JsonSerialize;
import trapx00.tagx00.exception.daoexception.IdDoesNotExistException;
import trapx00.tagx00.util.AnnotationUtil;
import trapx00.tagx00.util.PathUtil;

import java.io.*;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

@Service
public class FileServiceImpl<T extends Entity> implements FileService<T> {
    private final static String savePath = PathUtil.getDatabasePath();
    private final static String fileType = ".txt";

    /**
     * save the entity
     *
     * @param entity the entity object
     * @return the entity if success else return null
     */
    @Override
    public T saveTuple(T entity) {
        Class<? extends Entity> clazz = entity.getClass();
        String tableName = AnnotationUtil.getTableName(clazz);
        ArrayList<String> fileContent = new ArrayList<>();
        String id = AnnotationUtil.getKey(clazz);
        boolean isKeyAutoGenerated = AnnotationUtil.isKeyAutoGenerated(clazz);
        JSONObject json = JSONObject.fromObject(entity);

        //将json的key变为column的name,建立序列化文件
        Field[] fields = clazz.getDeclaredFields();
        ArrayList<String> columns = AnnotationUtil.getAllColumnName(clazz);
        Field idField = null;
        for (int i = 0; i < columns.size(); i++) {
            Object object = json.remove(fields[i].getName());
            if (fields[i].getAnnotation(Id.class) != null) {
                idField = fields[i];
            }
            if (fields[i].getAnnotation(JsonSerialize.class) != null) {
                try {
                    int serId = 0;
                    if (idField != null) {
                        idField.setAccessible(true);
                        serId = (int) idField.get(entity);
                    }
                    fields[i].setAccessible(true);
                    serId = serId == 0 ? 1 : serId;

                    FileOutputStream fileOut =
                            new FileOutputStream(PathUtil.getSerPath() + columns.get(i) + "_" + serId);
                    ObjectOutputStream out = new ObjectOutputStream(fileOut);
                    out.writeObject(fields[i].get(entity));
                    out.close();
                    fileOut.close();
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            } else {
                json.put(columns.get(i), object);
            }
        }

        int maxId = 0;
        try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(
                new FileInputStream(savePath + tableName + fileType)))) {
            boolean isUpdate = false;
            String jsonLine;
            while ((jsonLine = bufferedReader.readLine()) != null) {
                JSONObject jsonObject = JSONObject.fromObject(jsonLine);
                if (isKeyAutoGenerated && (Integer) jsonObject.get(id) > maxId) {
                    maxId = (Integer) jsonObject.get(id);
                }
                if (jsonObject.get(id).equals(json.get(id))) {
                    jsonLine = json.toString();
                    isUpdate = true;
                }
                fileContent.add(jsonLine);
            }
            if (!isUpdate) {
                if (isKeyAutoGenerated) {
                    json.element(id, maxId + 1);
                }
                fileContent.add(json.toString());
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        try (FileWriter writer = new FileWriter(savePath + tableName + fileType)) {
            writer.write("");
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try (FileWriter writer = new FileWriter(savePath + tableName + fileType, true)) {
            for (String tuple : fileContent) {
                writer.write(tuple);
                writer.write(System.lineSeparator());
                writer.flush();
            }
            return entity;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * find a entity
     *
     * @param info the key info to find
     * @return the entity
     */
    @Override
    public T findOne(String info, Class<T> clazz) {
        String methodName = new Exception().getStackTrace()[2].getMethodName();
        String columnName = methodName.split("By")[1];
        columnName = columnName.replaceFirst(columnName.substring(0, 1), columnName.substring(0, 1).toLowerCase());
        try {
            columnName = clazz.getDeclaredField(columnName).getAnnotation(Column.class).name();
        } catch (NoSuchFieldException e) {
            e.printStackTrace();
        }
        String tableName = AnnotationUtil.getTableName(clazz);

        try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(
                new FileInputStream(savePath + tableName + fileType)))) {
            String json;
            while ((json = bufferedReader.readLine()) != null) {
                JSONObject jsonObject = JSONObject.fromObject(json);
                if (jsonObject.get(columnName).toString().equals(info)) {
                    return fromJsonToObject(jsonObject, clazz);
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }

    @Override
    public void delete(String id, Class<T> clazz) {
        String tableName = AnnotationUtil.getTableName(clazz);
        String idName = AnnotationUtil.getKey(clazz);
        ArrayList<String> fileContent = new ArrayList<>();
        try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(
                new FileInputStream(savePath + tableName + fileType)))) {
            boolean isExist = false;
            String jsonLine;
            while ((jsonLine = bufferedReader.readLine()) != null) {
                JSONObject jsonObject = JSONObject.fromObject(jsonLine);
                if (jsonObject.get(idName).toString().equals(id)) {
                    isExist = true;
                } else {
                    fileContent.add(jsonLine);
                }
            }
            if (!isExist) {
                throw new IdDoesNotExistException();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        Field[] fields = clazz.getDeclaredFields();
        for (int i = 0; i < fields.length; i++) {
            if (fields[i].getAnnotation(JsonSerialize.class) != null) {
                String serFileName = PathUtil.getSerPath() + fields[i].getAnnotation(Column.class).name() + "_" + id;
                new File(serFileName).delete();
            }
        }

        try (FileWriter writer = new FileWriter(savePath + tableName + fileType)) {
            writer.write("");
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try (FileWriter writer = new FileWriter(savePath + tableName + fileType, true)) {
            for (String tuple : fileContent) {
                writer.write(tuple);
                writer.write(System.lineSeparator());
                writer.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public T[] findOnes(String info, Class<T> clazz) {
        String methodName = new Exception().getStackTrace()[2].getMethodName();
        String columnName = methodName.split("By")[1];
        columnName = columnName.replaceFirst(columnName.substring(0, 1), columnName.substring(0, 1).toLowerCase());
        try {
            columnName = clazz.getDeclaredField(columnName).getAnnotation(Column.class).name();
        } catch (NoSuchFieldException e) {
            e.printStackTrace();
        }
        String tableName = AnnotationUtil.getTableName(clazz);

        ArrayList<T> tArrayList = new ArrayList<>();
        try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(
                new FileInputStream(savePath + tableName + fileType)))) {
            String json;
            while ((json = bufferedReader.readLine()) != null) {
                JSONObject jsonObject = JSONObject.fromObject(json);
                if (jsonObject.get(columnName).toString().equals(info)) {
                    tArrayList.add(fromJsonToObject(jsonObject, clazz));
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return (T[]) tArrayList.toArray();
    }

    private T fromJsonToObject(JSONObject jsonObject, Class<T> clazz) {
        try {
            T t = clazz.newInstance();
            Field[] fields = clazz.getDeclaredFields();
            ArrayList<String> columns = AnnotationUtil.getAllColumnName(clazz);
            String idName = "";
            for (int i = 0; i < columns.size(); i++) {
                fields[i].setAccessible(true);
                ElementCollection elementCollection = fields[i].getAnnotation(ElementCollection.class);
                JsonSerialize jsonSerialize = fields[i].getAnnotation(JsonSerialize.class);
                if (fields[i].getAnnotation(Id.class) != null) {
                    idName = columns.get(i);
                }
                if (jsonSerialize != null) {
                    try {
                        int serId = 0;
                        if (idName != null && idName.length() > 0) {
                            serId = (int) jsonObject.get(idName);
                        }
                        fields[i].setAccessible(true);

                        FileInputStream fileIn = new FileInputStream(PathUtil.getSerPath() + columns.get(i) + "_" + serId);
                        ObjectInputStream in = new ObjectInputStream(fileIn);
                        Object serObject = in.readObject();
                        in.close();
                        fileIn.close();
                        fields[i].set(t, serObject);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (ClassNotFoundException e) {
                        e.printStackTrace();
                    }
                } else if (elementCollection != null) {
                    Class listClazz = elementCollection.targetClass();
                    JSONArray jsonArray = jsonObject.getJSONArray(columns.get(i));
                    List<?> list = JSONArray.toList(jsonArray, listClazz.newInstance(), new JsonConfig());
                    fields[i].set(t, list);
                } else {
                    fields[i].set(t, jsonObject.get(columns.get(i)));
                }
            }
            return t;
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
        return null;
    }
}
